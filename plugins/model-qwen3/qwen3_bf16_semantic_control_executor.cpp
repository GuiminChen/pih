#include "pih/model/qwen3_bf16_semantic_control_executor.h"

namespace pih {

Result<QwenBf16SemanticControlExecutor>
QwenBf16SemanticControlExecutor::Admit(
    QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
    std::uint32_t sequence_generation,
    const QwenBf16TapFixtureSequence& sequence,
    QwenBf16SequenceBackend& backend,
    QwenBf16CompletionIdentityProvider& completion) {
  auto digest = sequence.semantic_digest();
  if (!digest.ok()) return digest.status();
  auto lease = QwenKvSequenceLease::Admit(
      pool, owner_sequence_index, sequence_generation,
      QwenBf16TapFixtureSequence::kRequiredTokenCount);
  if (!lease.ok()) return lease.status();
  return QwenBf16SemanticControlExecutor(
      std::move(*lease), pool, sequence, backend, completion);
}

Status QwenBf16SemanticControlExecutor::run() {
  if (state_ != QwenBf16SemanticControlExecutorState::kReady) {
    return Status::FailedPrecondition(
        "Qwen semantic control executor is not ready");
  }
  state_ = QwenBf16SemanticControlExecutorState::kRunning;
  const auto fail = [this](Status status) {
    state_ = QwenBf16SemanticControlExecutorState::kPoisoned;
    return status;
  };
  for (std::size_t index = 0; index < sequence_.size(); ++index) {
    const auto input = sequence_[index];
    auto append = lease_.prepare_append(input.capture_position + 1U);
    if (!append.ok()) return fail(append.status());
    auto sampled = backend_->execute_and_read_token(
        input.tokens, input.first_position, lease_.block_table(), *append);
    if (!sampled.ok()) {
      const Status rollback = lease_.rollback_append(*append);
      return fail(rollback.ok() ? sampled.status() : rollback);
    }
    auto completion = completion_->last_completion_event();
    if (!completion.ok() || completion->handle == 0 ||
        completion->generation == 0) {
      const Status rollback = lease_.rollback_append(*append);
      if (!rollback.ok()) return fail(rollback);
      return fail(completion.ok()
                      ? Status::FailedPrecondition(
                            "Qwen semantic control completion is invalid")
                      : completion.status());
    }
    const Status committed = lease_.commit_append(*append);
    if (!committed.ok()) return fail(committed);
    const Status recorded = token_ledger_.commit(
        input.first_position, input.tokens, input.capture_position, *sampled);
    if (!recorded.ok()) return fail(recorded);
    last_completion_ = *completion;
  }
  state_ = QwenBf16SemanticControlExecutorState::kSealed;
  return Status::Ok();
}

Status QwenBf16SemanticControlExecutor::publish_token_semantics(
    QwenSemanticOutcomeRecorder& recorder) {
  if (state_ != QwenBf16SemanticControlExecutorState::kSealed) {
    return Status::FailedPrecondition(
        "Qwen semantic control token semantics are not publishable");
  }
  return token_ledger_.seal(recorder);
}

Result<QwenKvSemanticObservationPlan>
QwenBf16SemanticControlExecutor::make_kv_observation_plan(
    std::uint64_t backing_bytes) const {
  if (state_ != QwenBf16SemanticControlExecutorState::kSealed) {
    return Status::FailedPrecondition(
        "Qwen semantic control KV state is not observable");
  }
  const auto& table = lease_.block_table();
  auto states = pool_->project_prefix(
      table.descriptor().owner_sequence_index, table.reserved_handles(),
      table.descriptor().committed_tokens);
  if (!states.ok()) return states.status();
  return QwenKvSemanticObservationPlan::Create(table, *states, backing_bytes);
}

Status QwenBf16SemanticControlExecutor::release() {
  if (state_ != QwenBf16SemanticControlExecutorState::kSealed &&
      state_ != QwenBf16SemanticControlExecutorState::kPoisoned) {
    return Status::FailedPrecondition(
        "Qwen semantic control executor cannot release");
  }
  if (last_completion_.handle == 0 || last_completion_.generation == 0) {
    return Status::FailedPrecondition(
        "Qwen semantic control executor has no completion frontier");
  }
  const Status status = lease_.release(last_completion_);
  if (!status.ok()) return status;
  state_ = QwenBf16SemanticControlExecutorState::kReleased;
  return Status::Ok();
}

}  // namespace pih
