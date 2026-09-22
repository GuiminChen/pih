#include "pih/model/deepseek_prepared_boundary_operation.h"

#include <utility>

namespace pih {

Result<DeepSeekPreparedBoundaryOperation>
DeepSeekPreparedBoundaryOperation::Create(
    DeepSeekPipelineTransaction& transaction,
    DeepSeekNcclP2pManifest manifest, Buffer& buffer,
    std::uint64_t buffer_owner_id, std::uintptr_t context_identity,
    std::uintptr_t boundary_stream, DriverEventHandle completion_event,
    DeepSeekNcclOperationSequencer& sequencer,
    std::uint64_t submit_ns, std::uint64_t deadline_ns) {
  if (transaction.state() != DeepSeekPipelineTransactionState::kPrepared ||
      manifest.engine_epoch != transaction.descriptor().engine_epoch ||
      manifest.pipeline_plan_sequence !=
          transaction.descriptor().plan_sequence) {
    return Status::FailedPrecondition(
        "DeepSeek prepared boundary transaction identity is invalid");
  }
  auto operation = DeepSeekNcclP2pPlan::Create(manifest);
  if (!operation.ok()) return operation.status();
  auto boundary = DeepSeekNcclBoundaryLease::Create(
      manifest, buffer, buffer_owner_id, context_identity, boundary_stream);
  if (!boundary.ok()) return boundary.status();
  auto event = CompletionEventSlot::Create(completion_event, context_identity);
  if (!event.ok()) return event.status();
  auto frontier = CudaCompletionFrontier::Create(
      {manifest.engine_epoch, manifest.local_global_rank,
       manifest.pipeline_plan_sequence, CudaCompletionPhase::kCopy,
       manifest.operation_ordinal},
      manifest.operation_ordinal, submit_ns, deadline_ns);
  if (!frontier.ok()) return frontier.status();
  if (sequencer.poisoned() ||
      manifest.operation_ordinal < sequencer.next_ordinal()) {
    return Status::FailedPrecondition(
        "DeepSeek prepared boundary sequencer identity is invalid");
  }
  return DeepSeekPreparedBoundaryOperation(
      transaction,
      std::make_unique<DeepSeekNcclP2pPlan>(std::move(*operation)),
      std::make_unique<DeepSeekNcclBoundaryLease>(std::move(*boundary)),
      sequencer,
      std::make_unique<CompletionEventSlot>(std::move(*event)),
      std::make_unique<CudaCompletionFrontier>(std::move(*frontier)));
}

Status DeepSeekPreparedBoundaryOperation::ensure_executor() {
  if (executor_ != nullptr) return Status::Ok();
  if (transaction_->state() != DeepSeekPipelineTransactionState::kCommitted) {
    return Status::FailedPrecondition(
        "DeepSeek prepared boundary cannot execute before COMMIT");
  }
  auto executor = DeepSeekPipelineBoundaryExecutor::Create(
      *transaction_, *operation_, *boundary_, *sequencer_,
      *completion_event_, *completion_frontier_);
  if (!executor.ok()) {
    state_ = DeepSeekBoundaryExecutorState::kPoisoned;
    return executor.status();
  }
  executor_ = std::make_unique<DeepSeekPipelineBoundaryExecutor>(
      std::move(*executor));
  return Status::Ok();
}

Status DeepSeekPreparedBoundaryOperation::submit(
    DeepSeekNcclP2pBindingTarget& binding, DeepSeekNcclP2pDriver& nccl,
    CompletionEventDriver& event_driver) {
  const auto ready = ensure_executor();
  if (!ready.ok()) return ready;
  return executor_->submit(binding, nccl, event_driver);
}

Status DeepSeekPreparedBoundaryOperation::poll_issue(
    DeepSeekNcclP2pDriver& nccl, CompletionEventDriver& event_driver) {
  if (executor_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek prepared boundary was not submitted");
  }
  return executor_->poll_issue(nccl, event_driver);
}

Status DeepSeekPreparedBoundaryOperation::poll_completion(
    DeepSeekNcclP2pDriver& nccl, CompletionEventDriver& event_driver,
    CompletionEvidenceProvider& evidence) {
  if (executor_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek prepared boundary was not submitted");
  }
  return executor_->poll_completion(nccl, event_driver, evidence);
}

Status DeepSeekPreparedBoundaryOperation::expire(std::uint64_t now_ns) {
  if (executor_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek prepared boundary was not submitted");
  }
  return executor_->expire(now_ns);
}

DeepSeekBoundaryExecutorState DeepSeekPreparedBoundaryOperation::state()
    const noexcept {
  return executor_ == nullptr ? state_ : executor_->state();
}

}  // namespace pih
