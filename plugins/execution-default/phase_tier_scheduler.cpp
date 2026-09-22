#include "pih/scheduler/phase_tier_scheduler.h"

#include <algorithm>
#include <array>
#include <limits>

namespace pih {
namespace {
bool valid_phase(PackedTokenPhase phase) {
  return phase == PackedTokenPhase::kPrefill ||
         phase == PackedTokenPhase::kDecode ||
         phase == PackedTokenPhase::kVerify;
}

bool key_less(const SchedulerCandidate& left,
              const SchedulerCandidate& right) {
  if (left.last_service_plan_seq != right.last_service_plan_seq) {
    return left.last_service_plan_seq < right.last_service_plan_seq;
  }
  if (left.ready_enqueue_seq != right.ready_enqueue_seq) {
    return left.ready_enqueue_seq < right.ready_enqueue_seq;
  }
  return left.sequence_generation < right.sequence_generation;
}
}  // namespace

PhaseTierScheduler::PhaseTierScheduler(PhaseTierProfile profile)
    : profile_(profile),
      ordered_indices_(profile.maximum_candidates),
      included_indices_(profile.maximum_sequences_per_plan),
      results_(profile.maximum_candidates, SchedulerInfeasibility::kNone),
      scanned_(profile.maximum_candidates),
      selected_generations_(profile.maximum_sequences_per_plan),
      selected_token_counts_(profile.maximum_sequences_per_plan),
      plan_inputs_(profile.maximum_sequences_per_plan),
      metadata_inputs_(profile.maximum_sequences_per_plan) {}

Result<PhaseTierScheduler> PhaseTierScheduler::Create(
    PhaseTierProfile profile) {
  if (profile.maximum_candidates == 0 ||
      profile.maximum_sequences_per_plan == 0 ||
      profile.maximum_sequences_per_plan > profile.maximum_candidates ||
      profile.maximum_real_tokens_per_plan == 0 ||
      profile.scheduler_scan_limit < profile.maximum_candidates ||
      profile.max_consecutive_decode_rounds == 0 ||
      profile.prefill_starvation_threshold_ns <= 0) {
    return Status::InvalidArgument("phase-tier scheduler profile is invalid");
  }
  return PhaseTierScheduler(profile);
}

Result<SchedulerDecisionView> PhaseTierScheduler::select(
    const SchedulerSnapshot& snapshot) {
  if (snapshot.now_ns < 0 ||
      snapshot.candidates.size() > profile_.maximum_candidates ||
      snapshot.candidates.size() > profile_.scheduler_scan_limit ||
      generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return Status::InvalidArgument("scheduler snapshot exceeds its profile");
  }
  bool has_prefill = false;
  bool has_decode = false;
  bool has_protected = false;
  for (std::size_t candidate_index = 0;
       candidate_index < snapshot.candidates.size(); ++candidate_index) {
    const auto& candidate = snapshot.candidates[candidate_index];
    if (candidate.sequence_generation == 0 || candidate.ready_enqueue_seq == 0 ||
        candidate.real_token_count == 0 || !valid_phase(candidate.next_phase) ||
        candidate.prefill_eligible_since_ns < 0 ||
        candidate.prefill_eligible_since_ns > snapshot.now_ns ||
        candidate.infeasibility > SchedulerInfeasibility::kPlanTokenBound) {
      return Status::InvalidArgument("scheduler candidate is invalid");
    }
    for (std::size_t prior = 0; prior < candidate_index; ++prior) {
      if (snapshot.candidates[prior].sequence_generation ==
              candidate.sequence_generation ||
          snapshot.candidates[prior].ready_enqueue_seq ==
              candidate.ready_enqueue_seq) {
        return Status::InvalidArgument("scheduler candidate identity is duplicated");
      }
    }
    if (candidate.next_phase == PackedTokenPhase::kPrefill) {
      has_prefill = true;
      const auto age = snapshot.now_ns - candidate.prefill_eligible_since_ns;
      if (age >= profile_.prefill_starvation_threshold_ns) has_protected = true;
    } else {
      has_decode = true;
    }
  }

  std::array<SchedulerTier, 4> preferences{};
  std::size_t preference_count = 0;
  const auto append_unique = [&](SchedulerTier tier) {
    if (tier == SchedulerTier::kIdle) return;
    for (std::size_t i = 0; i < preference_count; ++i) {
      if (preferences[i] == tier) return;
    }
    preferences[preference_count++] = tier;
  };
  if (has_protected &&
      (!has_decode || !snapshot.last_successful_plan_was_protected_prefill)) {
    append_unique(SchedulerTier::kProtectedPrefill);
  }
  if (has_decode &&
      (snapshot.last_successful_plan_was_protected_prefill || !has_prefill ||
       snapshot.consecutive_decode_rounds <
           profile_.max_consecutive_decode_rounds)) {
    append_unique(SchedulerTier::kDecodeFamily);
  }
  if (has_prefill) append_unique(SchedulerTier::kNormalPrefill);
  if (has_decode) append_unique(SchedulerTier::kDecodeFamily);

  std::fill_n(scanned_.begin(), snapshot.candidates.size(), 0);
  std::fill_n(results_.begin(), snapshot.candidates.size(),
              SchedulerInfeasibility::kNone);
  std::uint32_t scanned_count = 0;
  std::uint32_t included_count = 0;
  std::uint32_t total_tokens = 0;
  SchedulerTier selected_tier = SchedulerTier::kIdle;
  PackedTokenPhase selected_phase = PackedTokenPhase::kPrefill;

  for (std::size_t p = 0; p < preference_count && included_count == 0; ++p) {
    const auto tier = preferences[p];
    std::size_t ordered_count = 0;
    for (std::uint32_t index = 0; index < snapshot.candidates.size(); ++index) {
      const auto& candidate = snapshot.candidates[index];
      bool belongs = false;
      if (tier == SchedulerTier::kProtectedPrefill) {
        belongs = candidate.next_phase == PackedTokenPhase::kPrefill &&
                  snapshot.now_ns - candidate.prefill_eligible_since_ns >=
                      profile_.prefill_starvation_threshold_ns;
      } else if (tier == SchedulerTier::kNormalPrefill) {
        belongs = candidate.next_phase == PackedTokenPhase::kPrefill;
      } else {
        belongs = candidate.next_phase != PackedTokenPhase::kPrefill;
      }
      if (belongs && !scanned_[index]) ordered_indices_[ordered_count++] = index;
    }
    std::sort(ordered_indices_.begin(), ordered_indices_.begin() + ordered_count,
              [&](std::uint32_t left, std::uint32_t right) {
                return key_less(snapshot.candidates[left],
                                snapshot.candidates[right]);
              });
    PackedTokenPhase tier_phase = PackedTokenPhase::kPrefill;
    if (tier == SchedulerTier::kDecodeFamily && ordered_count != 0) {
      tier_phase = snapshot.candidates[ordered_indices_[0]].next_phase;
    }
    for (std::size_t ordinal = 0; ordinal < ordered_count; ++ordinal) {
      const auto index = ordered_indices_[ordinal];
      const auto& candidate = snapshot.candidates[index];
      if (candidate.next_phase != tier_phase) continue;
      scanned_[index] = 1;
      ++scanned_count;
      if (candidate.infeasibility != SchedulerInfeasibility::kNone) {
        results_[index] = candidate.infeasibility;
        continue;
      }
      if (included_count == profile_.maximum_sequences_per_plan ||
          candidate.real_token_count >
              profile_.maximum_real_tokens_per_plan - total_tokens) {
        results_[index] = SchedulerInfeasibility::kPlanTokenBound;
        continue;
      }
      included_indices_[included_count++] = index;
      selected_generations_[included_count - 1] = candidate.sequence_generation;
      selected_token_counts_[included_count - 1] = candidate.real_token_count;
      total_tokens += candidate.real_token_count;
    }
    if (included_count != 0) {
      selected_tier = tier;
      selected_phase = tier_phase;
    }
  }

  ++generation_;
  current_candidate_count_ = static_cast<std::uint32_t>(snapshot.candidates.size());
  current_included_count_ = included_count;
  current_total_tokens_ = total_tokens;
  current_tier_ = selected_tier;
  current_phase_ = selected_phase;
  return SchedulerDecisionView{
      generation_, selected_tier, selected_phase,
      std::span(included_indices_).first(included_count),
      std::span(results_).first(snapshot.candidates.size()), scanned_count,
      total_tokens};
}

Result<PackedTokenPlan> PhaseTierScheduler::assemble_current_plan(
    std::uint64_t decision_generation, ScheduledPlanIdentity identity,
    std::span<const ScheduledSequenceState> candidate_states) {
  if (decision_generation == 0 || decision_generation != generation_ ||
      current_tier_ == SchedulerTier::kIdle || current_included_count_ == 0 ||
      candidate_states.size() != current_candidate_count_) {
    return Status::InvalidArgument("scheduler decision is stale or idle");
  }
  std::uint64_t checked_total = 0;
  for (std::uint32_t ordinal = 0; ordinal < current_included_count_; ++ordinal) {
    const auto candidate_index = included_indices_[ordinal];
    if (candidate_index >= candidate_states.size()) {
      return Status::InvalidArgument("scheduler decision index is invalid");
    }
    const auto& state = candidate_states[candidate_index];
    if (state.sequence_generation != selected_generations_[ordinal] ||
        state.state_generation == 0) {
      return Status::InvalidArgument("scheduled sequence state drifted");
    }
    checked_total += selected_token_counts_[ordinal];
    plan_inputs_[ordinal] = PackedSequenceInput{
        state.sequence_generation, state.committed_start_position,
        selected_token_counts_[ordinal], state.state_generation,
        state.input_digest};
  }
  if (checked_total != current_total_tokens_) {
    return Status::Internal("scheduler selected-token ledger drifted");
  }
  return PackedTokenPlan::Create(
      identity.epoch, identity.plan_sequence, current_phase_,
      std::move(identity.profile_revision), identity.resource_vector_hash,
      std::span(plan_inputs_).first(current_included_count_),
      identity.execution_bucket_tokens,
      PackedTokenPlanLimits{profile_.maximum_sequences_per_plan,
                            profile_.maximum_real_tokens_per_plan,
                            identity.maximum_execution_bucket_tokens});
}

Result<PackedTokenMetadataView>
PhaseTierScheduler::materialize_current_metadata(
    std::uint64_t decision_generation, const PackedTokenPlan& plan,
    std::span<const ScheduledSequenceTokens> candidate_tokens,
    PackedTokenMetadataArena& arena) {
  if (decision_generation == 0 || decision_generation != generation_ ||
      current_tier_ == SchedulerTier::kIdle || current_included_count_ == 0 ||
      candidate_tokens.size() != current_candidate_count_ ||
      plan.phase() != current_phase_ ||
      plan.sequence_count() != current_included_count_ ||
      plan.total_real_tokens() != current_total_tokens_) {
    return Status::InvalidArgument("packed metadata decision identity drifted");
  }
  for (std::uint32_t ordinal = 0; ordinal < current_included_count_; ++ordinal) {
    const auto candidate_index = included_indices_[ordinal];
    if (candidate_index >= candidate_tokens.size() ||
        candidate_tokens[candidate_index].sequence_generation !=
            selected_generations_[ordinal] ||
        candidate_tokens[candidate_index].token_ids.size() !=
            selected_token_counts_[ordinal] ||
        plan.ordered_sequence_generations()[ordinal] !=
            selected_generations_[ordinal] ||
        plan.real_token_counts()[ordinal] != selected_token_counts_[ordinal]) {
      return Status::InvalidArgument("scheduled token payload drifted");
    }
    metadata_inputs_[ordinal] = PackedSequenceTokens{
        candidate_tokens[candidate_index].sequence_generation,
        candidate_tokens[candidate_index].token_ids,
        candidate_tokens[candidate_index].produces_logits};
  }
  return arena.materialize(
      plan, std::span(metadata_inputs_).first(current_included_count_));
}

}  // namespace pih
