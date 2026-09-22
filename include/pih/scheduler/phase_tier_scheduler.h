#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"
#include "pih/model/packed_token_metadata_arena.h"
#include "pih/model/packed_token_plan.h"

namespace pih {

enum class SchedulerTier : std::uint8_t {
  kIdle = 0,
  kProtectedPrefill = 1,
  kDecodeFamily = 2,
  kNormalPrefill = 3,
};

enum class SchedulerInfeasibility : std::uint8_t {
  kNone = 0,
  kPersistentResource = 1,
  kWorkspace = 2,
  kBoundaryCredit = 3,
  kExpertPaging = 4,
  kOutputCredit = 5,
  kPlanTokenBound = 6,
};

struct SchedulerCandidate final {
  std::uint64_t sequence_generation;
  PackedTokenPhase next_phase;
  std::uint64_t last_service_plan_seq;
  std::uint64_t ready_enqueue_seq;
  std::int64_t prefill_eligible_since_ns;
  std::uint32_t real_token_count;
  SchedulerInfeasibility infeasibility;
};

struct PhaseTierProfile final {
  std::uint32_t maximum_candidates;
  std::uint32_t maximum_sequences_per_plan;
  std::uint32_t maximum_real_tokens_per_plan;
  std::uint32_t scheduler_scan_limit;
  std::uint32_t max_consecutive_decode_rounds;
  std::int64_t prefill_starvation_threshold_ns;
};

struct SchedulerSnapshot final {
  std::int64_t now_ns;
  std::uint32_t consecutive_decode_rounds;
  bool last_successful_plan_was_protected_prefill;
  std::span<const SchedulerCandidate> candidates;
};

struct SchedulerDecisionView final {
  std::uint64_t generation;
  SchedulerTier tier;
  PackedTokenPhase phase;
  std::span<const std::uint32_t> included_candidate_indices;
  std::span<const SchedulerInfeasibility> candidate_results;
  std::uint32_t scanned_count;
  std::uint32_t total_real_tokens;
};

struct ScheduledSequenceState final {
  std::uint64_t sequence_generation;
  std::uint64_t committed_start_position;
  std::uint64_t state_generation;
  Sha256Digest input_digest;
};

struct ScheduledPlanIdentity final {
  std::uint64_t epoch;
  std::uint64_t plan_sequence;
  std::string profile_revision;
  Sha256Digest resource_vector_hash;
  std::uint32_t execution_bucket_tokens;
  std::uint32_t maximum_execution_bucket_tokens;
};

struct ScheduledSequenceTokens final {
  std::uint64_t sequence_generation;
  std::span<const std::uint32_t> token_ids;
  bool produces_logits;
};

class PhaseTierScheduler final {
 public:
  static constexpr std::string_view kPolicyAbi =
      "phase_tier_round_robin_v1";

  static Result<PhaseTierScheduler> Create(PhaseTierProfile profile);
  Result<SchedulerDecisionView> select(const SchedulerSnapshot& snapshot);
  Result<PackedTokenPlan> assemble_current_plan(
      std::uint64_t decision_generation, ScheduledPlanIdentity identity,
      std::span<const ScheduledSequenceState> candidate_states);
  Result<PackedTokenMetadataView> materialize_current_metadata(
      std::uint64_t decision_generation, const PackedTokenPlan& plan,
      std::span<const ScheduledSequenceTokens> candidate_tokens,
      PackedTokenMetadataArena& arena);

 private:
  explicit PhaseTierScheduler(PhaseTierProfile profile);

  PhaseTierProfile profile_{};
  std::uint64_t generation_ = 0;
  std::vector<std::uint32_t> ordered_indices_;
  std::vector<std::uint32_t> included_indices_;
  std::vector<SchedulerInfeasibility> results_;
  std::vector<std::uint8_t> scanned_;
  std::vector<std::uint64_t> selected_generations_;
  std::vector<std::uint32_t> selected_token_counts_;
  std::vector<PackedSequenceInput> plan_inputs_;
  std::vector<PackedSequenceTokens> metadata_inputs_;
  std::uint32_t current_candidate_count_ = 0;
  std::uint32_t current_included_count_ = 0;
  std::uint32_t current_total_tokens_ = 0;
  SchedulerTier current_tier_ = SchedulerTier::kIdle;
  PackedTokenPhase current_phase_ = PackedTokenPhase::kPrefill;
};

}  // namespace pih
