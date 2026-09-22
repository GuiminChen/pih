#include "pih/model/qwen3_bf16_tap_suite_executor.h"

namespace pih {

Result<QwenBf16TapSuiteExecutor> QwenBf16TapSuiteExecutor::Admit(
    QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
    std::uint32_t sequence_generation,
    const QwenBf16TapSuitePlan& suite,
    const QwenBf16TapFixtureSequence& sequence,
    std::uint64_t suite_generation,
    std::uint64_t first_fixture_generation,
    QwenBf16TapFixtureBackend& backend) {
  auto expected_sequence_digest = sequence.semantic_digest();
  if (!expected_sequence_digest.ok() ||
      sequence.size() != suite.size()) {
    return Status::InvalidArgument(
        "Qwen tap suite executor input is invalid");
  }
  auto run = QwenBf16TapSuiteRun::Create(
      suite, suite_generation, first_fixture_generation);
  if (!run.ok()) return run.status();
  auto lease = QwenKvSequenceLease::Admit(
      pool, owner_sequence_index, sequence_generation,
      QwenBf16TapFixtureSequence::kRequiredTokenCount);
  if (!lease.ok()) return lease.status();
  return QwenBf16TapSuiteExecutor(
      std::move(*lease), pool, suite, sequence, std::move(*run),
      first_fixture_generation, backend);
}

Result<QwenBf16TapSuiteRunReceipt> QwenBf16TapSuiteExecutor::run() {
  if (state_ != QwenBf16TapSuiteExecutorState::kReady) {
    return Status::FailedPrecondition("Qwen tap suite executor is not ready");
  }
  state_ = QwenBf16TapSuiteExecutorState::kRunning;
  const auto fail = [this](Status status)
      -> Result<QwenBf16TapSuiteRunReceipt> {
    state_ = QwenBf16TapSuiteExecutorState::kPoisoned;
    return status;
  };
  for (std::size_t index = 0; index < suite_.size(); ++index) {
    const auto input = sequence_[index];
    auto append = lease_.prepare_append(
        input.capture_position + 1U);
    if (!append.ok()) return fail(append.status());
    auto projected_slots = pool_->project_prefix(
        lease_.block_table().descriptor().owner_sequence_index,
        lease_.block_table().reserved_handles(),
        append->target_committed_tokens);
    if (!projected_slots.ok()) return fail(projected_slots.status());
    auto executed = backend_->execute_and_capture(
        input, suite_[index].taps, lease_.block_table(), *append,
        *projected_slots,
        first_fixture_generation_ + index);
    if (!executed.ok()) {
      const Status rolled_back = lease_.rollback_append(*append);
      return fail(rolled_back.ok() ? executed.status() : rolled_back);
    }
    if (executed->completion.handle == 0 ||
        executed->completion.generation == 0) {
      const Status rolled_back = lease_.rollback_append(*append);
      return fail(rolled_back.ok()
                      ? Status::FailedPrecondition(
                            "Qwen tap fixture completion is invalid")
                      : rolled_back);
    }
    const Status recorded = run_.record(index, executed->tap_run);
    if (!recorded.ok()) {
      const Status rolled_back = lease_.rollback_append(*append);
      return fail(rolled_back.ok() ? recorded : rolled_back);
    }
    const Status committed = lease_.commit_append(*append);
    if (!committed.ok()) return fail(committed);
    const Status token_recorded = token_ledger_.commit(
        input.first_position, input.tokens, input.capture_position,
        executed->sampled_token);
    if (!token_recorded.ok()) return fail(token_recorded);
    last_completion_ = executed->completion;
  }
  auto receipt = run_.seal();
  if (!receipt.ok()) return fail(receipt.status());
  state_ = QwenBf16TapSuiteExecutorState::kSealed;
  return receipt;
}

Status QwenBf16TapSuiteExecutor::publish_token_semantics(
    QwenSemanticOutcomeRecorder& recorder) {
  if (state_ != QwenBf16TapSuiteExecutorState::kSealed) {
    return Status::FailedPrecondition(
        "Qwen tap suite token semantics are not publishable");
  }
  return token_ledger_.seal(recorder);
}

Result<QwenKvSemanticObservationPlan>
QwenBf16TapSuiteExecutor::make_kv_observation_plan(
    std::uint64_t backing_bytes) const {
  if (state_ != QwenBf16TapSuiteExecutorState::kSealed) {
    return Status::FailedPrecondition(
        "Qwen tap suite KV semantics are not observable");
  }
  const auto& table = lease_.block_table();
  auto states = pool_->project_prefix(
      table.descriptor().owner_sequence_index, table.reserved_handles(),
      table.descriptor().committed_tokens);
  if (!states.ok()) return states.status();
  return QwenKvSemanticObservationPlan::Create(
      table, *states, backing_bytes);
}

Status QwenBf16TapSuiteExecutor::release() {
  if (state_ != QwenBf16TapSuiteExecutorState::kSealed &&
      state_ != QwenBf16TapSuiteExecutorState::kPoisoned) {
    return Status::FailedPrecondition(
        "Qwen tap suite executor cannot release in its current state");
  }
  if (last_completion_.handle == 0 || last_completion_.generation == 0) {
    return Status::FailedPrecondition(
        "Qwen tap suite executor has no completion frontier");
  }
  const Status released = lease_.release(last_completion_);
  if (!released.ok()) return released;
  state_ = QwenBf16TapSuiteExecutorState::kReleased;
  return Status::Ok();
}

}  // namespace pih
