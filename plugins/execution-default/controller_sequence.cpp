#include "pih/scheduler/controller_sequence.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {
ControllerSequenceState prepared_state(PackedTokenPhase phase) {
  switch (phase) {
    case PackedTokenPhase::kPrefill: return ControllerSequenceState::kPrefillPrepared;
    case PackedTokenPhase::kDecode: return ControllerSequenceState::kDecodePrepared;
    case PackedTokenPhase::kVerify: return ControllerSequenceState::kVerifyPrepared;
  }
  return ControllerSequenceState::kFailed;
}
ControllerSequenceState committed_state(PackedTokenPhase phase) {
  switch (phase) {
    case PackedTokenPhase::kPrefill: return ControllerSequenceState::kPrefillCommitted;
    case PackedTokenPhase::kDecode: return ControllerSequenceState::kDecodeCommitted;
    case PackedTokenPhase::kVerify: return ControllerSequenceState::kVerifyCommitted;
  }
  return ControllerSequenceState::kFailed;
}
ControllerSequenceState inflight_state(PackedTokenPhase phase) {
  switch (phase) {
    case PackedTokenPhase::kPrefill: return ControllerSequenceState::kPrefillInFlight;
    case PackedTokenPhase::kDecode: return ControllerSequenceState::kDecodeInFlight;
    case PackedTokenPhase::kVerify: return ControllerSequenceState::kVerifyInFlight;
  }
  return ControllerSequenceState::kFailed;
}
}  // namespace

Result<ControllerSequence> ControllerSequence::Admit(
    std::uint64_t sequence_generation, std::uint64_t prompt_token_count,
    std::uint64_t initial_state_generation) {
  if (sequence_generation == 0 || prompt_token_count == 0 ||
      initial_state_generation == 0) {
    return Status::InvalidArgument("controller sequence identity is invalid");
  }
  return ControllerSequence(sequence_generation, prompt_token_count,
                            initial_state_generation);
}

bool ControllerSequence::terminal(ControllerSequenceState state) noexcept {
  return state == ControllerSequenceState::kCancelled ||
         state == ControllerSequenceState::kCompleted ||
         state == ControllerSequenceState::kFailed;
}

bool ControllerSequence::prepared() const noexcept {
  return state_ == ControllerSequenceState::kPrefillPrepared ||
         state_ == ControllerSequenceState::kDecodePrepared ||
         state_ == ControllerSequenceState::kVerifyPrepared;
}

bool ControllerSequence::committed_or_in_flight() const noexcept {
  return state_ == ControllerSequenceState::kPrefillCommitted ||
         state_ == ControllerSequenceState::kPrefillInFlight ||
         state_ == ControllerSequenceState::kDecodeCommitted ||
         state_ == ControllerSequenceState::kDecodeInFlight ||
         state_ == ControllerSequenceState::kVerifyCommitted ||
         state_ == ControllerSequenceState::kVerifyInFlight;
}

void ControllerSequence::clear_plan() noexcept {
  active_plan_sequence_ = 0;
  active_real_token_count_ = 0;
}

Status ControllerSequence::prepare(std::uint64_t plan_sequence,
                                   PackedTokenPhase phase,
                                   std::uint32_t real_token_count) {
  const bool ready = state_ == ControllerSequenceState::kQueuedPrefill ||
                     state_ == ControllerSequenceState::kReadyDecode;
  const bool phase_matches =
      (state_ == ControllerSequenceState::kQueuedPrefill &&
       phase == PackedTokenPhase::kPrefill) ||
      (state_ == ControllerSequenceState::kReadyDecode &&
       phase == next_decode_phase_);
  if (!ready || !phase_matches || plan_sequence == 0 ||
      real_token_count == 0 || active_plan_sequence_ != 0) {
    return Status::FailedPrecondition("sequence cannot prepare this plan");
  }
  if (phase == PackedTokenPhase::kPrefill) {
    auto end = checked_add_u64(model_processed_length_, real_token_count);
    if (!end.ok() || *end > prompt_token_count_) {
      return Status::InvalidArgument("prefill plan exceeds prompt partition");
    }
  } else if (phase == PackedTokenPhase::kDecode && real_token_count != 1) {
    return Status::InvalidArgument("decode plan must contain one real token");
  } else if (phase == PackedTokenPhase::kVerify && real_token_count > 5) {
    return Status::InvalidArgument("verify plan exceeds the V1 draft bound");
  }
  active_plan_sequence_ = plan_sequence;
  active_real_token_count_ = real_token_count;
  active_phase_ = phase;
  state_ = prepared_state(phase);
  return Status::Ok();
}

Status ControllerSequence::commit(std::uint64_t plan_sequence) {
  if (!prepared() || plan_sequence != active_plan_sequence_) {
    return Status::FailedPrecondition("sequence plan is not prepared");
  }
  state_ = committed_state(active_phase_);
  return Status::Ok();
}

Status ControllerSequence::mark_in_flight(std::uint64_t plan_sequence) {
  if (!committed_or_in_flight() || state_ == inflight_state(active_phase_) ||
      plan_sequence != active_plan_sequence_) {
    return Status::FailedPrecondition("sequence plan is not committed");
  }
  state_ = inflight_state(active_phase_);
  return Status::Ok();
}

Status ControllerSequence::abort_prepared(std::uint64_t plan_sequence) {
  if (!prepared() || plan_sequence != active_plan_sequence_) {
    return Status::FailedPrecondition("only a prepared plan can be aborted");
  }
  state_ = active_phase_ == PackedTokenPhase::kPrefill
               ? ControllerSequenceState::kQueuedPrefill
               : ControllerSequenceState::kReadyDecode;
  clear_plan();
  return Status::Ok();
}

Status ControllerSequence::validate_completion(
    const ControllerPlanCompletion& completion) const {
  const bool cancelled_committed =
      state_ == ControllerSequenceState::kCancelRequested &&
      active_plan_sequence_ != 0 &&
      (state_before_cancel_ == ControllerSequenceState::kPrefillCommitted ||
       state_before_cancel_ == ControllerSequenceState::kPrefillInFlight ||
       state_before_cancel_ == ControllerSequenceState::kDecodeCommitted ||
       state_before_cancel_ == ControllerSequenceState::kDecodeInFlight ||
       state_before_cancel_ == ControllerSequenceState::kVerifyCommitted ||
       state_before_cancel_ == ControllerSequenceState::kVerifyInFlight);
  const bool normally_in_flight = state_ == inflight_state(active_phase_);
  if ((!normally_in_flight && !cancelled_committed) ||
      completion.plan_sequence != active_plan_sequence_ ||
      completion.sequence_generation != sequence_generation_ ||
      completion.state_generation <= state_generation_ ||
      completion.committed_model_processed_length < model_processed_length_ ||
      completion.accepted_completion_count < accepted_completion_count_ ||
      (completion.next_decode_phase != PackedTokenPhase::kDecode &&
       completion.next_decode_phase != PackedTokenPhase::kVerify)) {
    return Status::FailedPrecondition("plan completion identity is invalid");
  }
  if (active_phase_ == PackedTokenPhase::kPrefill) {
    auto expected = checked_add_u64(model_processed_length_, active_real_token_count_);
    if (!expected.ok() || completion.committed_model_processed_length != *expected ||
        *expected > prompt_token_count_) {
      return Status::InvalidArgument("prefill completion cut is invalid");
    }
  } else {
    const auto maximum = checked_add_u64(model_processed_length_,
                                         active_real_token_count_);
    if (!maximum.ok() ||
        completion.committed_model_processed_length > *maximum) {
      return Status::InvalidArgument("decode-family state cut is invalid");
    }
  }
  if (completion.accepted_completion_count != 0) {
    auto invariant = checked_add_u64(
        prompt_token_count_, completion.accepted_completion_count - 1);
    if (!invariant.ok() ||
        completion.committed_model_processed_length != *invariant) {
      return Status::InvalidArgument("K=P+A-1 ledger invariant failed");
    }
  } else if (completion.committed_model_processed_length >= prompt_token_count_) {
    return Status::InvalidArgument("final prefill must publish its first token");
  }
  return Status::Ok();
}

Status ControllerSequence::complete(const ControllerPlanCompletion& completion) {
  const auto validation = validate_completion(completion);
  if (!validation.ok()) return validation;
  const bool cancelled_committed =
      state_ == ControllerSequenceState::kCancelRequested;
  model_processed_length_ = completion.committed_model_processed_length;
  accepted_completion_count_ = completion.accepted_completion_count;
  state_generation_ = completion.state_generation;
  next_decode_phase_ = completion.next_decode_phase;
  clear_plan();
  if (cancelled_committed) {
    state_ = ControllerSequenceState::kDraining;
  } else if (completion.terminal) {
    state_ = ControllerSequenceState::kDraining;
  } else if (model_processed_length_ < prompt_token_count_) {
    state_ = ControllerSequenceState::kQueuedPrefill;
  } else {
    state_ = ControllerSequenceState::kReadyDecode;
  }
  return Status::Ok();
}

Status ControllerSequence::request_cancel() {
  if (terminal(state_) || state_ == ControllerSequenceState::kDraining) {
    return Status::FailedPrecondition("terminal sequence cannot be cancelled");
  }
  if (state_ == ControllerSequenceState::kCancelRequested) return Status::Ok();
  state_before_cancel_ = state_;
  drain_is_cancel_ = true;
  state_ = ControllerSequenceState::kCancelRequested;
  return Status::Ok();
}

Status ControllerSequence::begin_draining() {
  if (state_ != ControllerSequenceState::kCancelRequested ||
      (active_plan_sequence_ != 0 &&
       state_before_cancel_ != ControllerSequenceState::kPrefillPrepared &&
       state_before_cancel_ != ControllerSequenceState::kDecodePrepared &&
       state_before_cancel_ != ControllerSequenceState::kVerifyPrepared)) {
    return Status::FailedPrecondition("sequence still has committed work");
  }
  clear_plan();
  state_ = ControllerSequenceState::kDraining;
  return Status::Ok();
}

Status ControllerSequence::finish_cancel() {
  if (state_ != ControllerSequenceState::kDraining || !drain_is_cancel_) {
    return Status::FailedPrecondition("sequence is not draining");
  }
  state_ = ControllerSequenceState::kCancelled;
  return Status::Ok();
}

Status ControllerSequence::finish_drain() {
  if (state_ != ControllerSequenceState::kDraining ||
      drain_is_cancel_) {
    return Status::FailedPrecondition("sequence is not terminal-draining");
  }
  state_ = ControllerSequenceState::kCompleted;
  return Status::Ok();
}

Status ControllerSequence::fail() {
  if (terminal(state_)) {
    return Status::FailedPrecondition("sequence is already terminal");
  }
  clear_plan();
  state_ = ControllerSequenceState::kFailed;
  return Status::Ok();
}

Result<PackedTokenPhase> ControllerSequence::ready_phase() const {
  if (state_ == ControllerSequenceState::kQueuedPrefill) {
    return PackedTokenPhase::kPrefill;
  }
  if (state_ == ControllerSequenceState::kReadyDecode) {
    return next_decode_phase_;
  }
  return Status::FailedPrecondition("sequence is not at a schedulable boundary");
}

Result<std::uint32_t> ControllerSequence::next_real_token_count(
    std::uint32_t maximum_prefill_chunk_tokens,
    std::uint32_t verify_token_count) const {
  if (maximum_prefill_chunk_tokens == 0) {
    return Status::InvalidArgument("prefill chunk bound must be positive");
  }
  auto phase = ready_phase();
  if (!phase.ok()) return phase.status();
  if (*phase == PackedTokenPhase::kPrefill) {
    if (verify_token_count != 0 || model_processed_length_ >= prompt_token_count_) {
      return Status::InvalidArgument("prefill scheduling input is inconsistent");
    }
    const auto remaining = prompt_token_count_ - model_processed_length_;
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(remaining, maximum_prefill_chunk_tokens));
  }
  if (*phase == PackedTokenPhase::kDecode) {
    if (verify_token_count != 0) {
      return Status::InvalidArgument("decode cannot carry verify tokens");
    }
    return 1U;
  }
  if (verify_token_count == 0 || verify_token_count > 5) {
    return Status::InvalidArgument("verify token count is outside the V1 bound");
  }
  return verify_token_count;
}

Result<bool> ControllerSequence::next_plan_produces_logits(
    std::uint32_t maximum_prefill_chunk_tokens,
    std::uint32_t verify_token_count) const {
  auto count = next_real_token_count(maximum_prefill_chunk_tokens,
                                     verify_token_count);
  if (!count.ok()) return count.status();
  auto phase = ready_phase();
  if (!phase.ok()) return phase.status();
  if (*phase != PackedTokenPhase::kPrefill) return true;
  return model_processed_length_ + *count == prompt_token_count_;
}

}  // namespace pih
