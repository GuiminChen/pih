#include "pih/model/deepseek_dspark_controller_publish_operations.h"

#include "pih/core/checked_math.h"

namespace pih {
Result<DeepSeekDsparkControllerPublishOperations>
DeepSeekDsparkControllerPublishOperations::Create(
    ControllerSequence& sequence, std::uint64_t sequence_generation,
    PackedTokenPhase next_decode_phase) {
  if (sequence_generation == 0 ||
      sequence.sequence_generation() != sequence_generation ||
      sequence.state() != ControllerSequenceState::kVerifyInFlight ||
      (next_decode_phase != PackedTokenPhase::kDecode &&
       next_decode_phase != PackedTokenPhase::kVerify)) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark controller publisher is not creatable");
  }
  DeepSeekDsparkControllerPublishOperations value;
  value.sequence_ = &sequence;
  value.sequence_generation_ = sequence_generation;
  value.next_decode_phase_ = next_decode_phase;
  return value;
}
Status DeepSeekDsparkControllerPublishOperations::validate_engine_healthy() {
  if (sequence_->state() != ControllerSequenceState::kVerifyInFlight ||
      sequence_->sequence_generation() != sequence_generation_)
    return Status::FailedPrecondition(
        "DeepSeek DSpark controller sequence changed during publish");
  return Status::Ok();
}
Status DeepSeekDsparkControllerPublishOperations::validate_decision(
    const DeepSeekDsparkStateCutDecision& decision) const {
  auto processed = checked_add_u64(sequence_->model_processed_length(),
                                   decision.processed_delta);
  auto accepted = checked_add_u64(sequence_->accepted_completion_count(),
                                  decision.retained_record_count);
  if (!processed.ok()) return processed.status();
  if (!accepted.ok()) return accepted.status();
  return sequence_->validate_completion({
      decision.plan_sequence, sequence_generation_, *processed, *accepted,
      decision.new_generation, next_decode_phase_, decision.terminal_drain});
}
Status DeepSeekDsparkControllerPublishOperations::publish_ledger_and_output(
    const DeepSeekDsparkStateCutDecision& decision) {
  if (ledger_published_ ||
      decision.old_generation != sequence_->state_generation())
    return Status::FailedPrecondition(
        "DeepSeek DSpark ledger is not publishable");
  auto validation = validate_decision(decision);
  if (!validation.ok()) return validation;
  auto processed = checked_add_u64(sequence_->model_processed_length(),
                                   decision.processed_delta);
  auto accepted = checked_add_u64(sequence_->accepted_completion_count(),
                                  decision.retained_record_count);
  if (!processed.ok()) return processed.status();
  if (!accepted.ok()) return accepted.status();
  auto status = sequence_->complete({
      decision.plan_sequence, sequence_generation_, *processed, *accepted,
      decision.new_generation, next_decode_phase_, decision.terminal_drain});
  if (status.ok()) ledger_published_ = true;
  return status;
}
}  // namespace pih
