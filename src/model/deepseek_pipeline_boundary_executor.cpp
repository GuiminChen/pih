#include "pih/model/deepseek_pipeline_boundary_executor.h"

namespace pih {

Result<DeepSeekPipelineBoundaryExecutor>
DeepSeekPipelineBoundaryExecutor::Create(
    const DeepSeekPipelineTransaction& transaction,
    DeepSeekNcclP2pPlan& operation,
    const DeepSeekNcclBoundaryLease& boundary,
    DeepSeekNcclOperationSequencer& sequencer,
    CompletionEventSlot& completion_event,
    CudaCompletionFrontier& completion_frontier) {
  if (transaction.state() != DeepSeekPipelineTransactionState::kCommitted ||
      operation.state() != DeepSeekNcclP2pState::kPlanned ||
      boundary.bytes() != operation.wire_bytes() ||
      completion_event.context_identity() != operation.manifest().context_identity ||
      completion_frontier.event_generation() !=
          operation.manifest().operation_ordinal ||
      completion_frontier.key().epoch != operation.manifest().engine_epoch ||
      completion_frontier.key().plan_generation !=
          operation.manifest().pipeline_plan_sequence ||
      sequencer.poisoned() ||
      sequencer.next_ordinal() != operation.manifest().operation_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek boundary executor identities do not join");
  }
  return DeepSeekPipelineBoundaryExecutor(
      transaction, operation, boundary, sequencer, completion_event,
      completion_frontier);
}

Status DeepSeekPipelineBoundaryExecutor::poison(Status status) {
  state_ = DeepSeekBoundaryExecutorState::kPoisoned;
  sequencer_->poison_epoch();
  return status.ok() ? Status::Internal("DeepSeek boundary executor poisoned")
                     : status;
}

Status DeepSeekPipelineBoundaryExecutor::record_event(
    CompletionEventDriver& event_driver) {
  const auto status = operation_->record_completion(
      *completion_event_, event_driver, boundary_->stream());
  if (!status.ok()) return poison(status);
  state_ = DeepSeekBoundaryExecutorState::kDeviceInFlight;
  return Status::Ok();
}

Status DeepSeekPipelineBoundaryExecutor::submit(
    DeepSeekNcclP2pBindingTarget& binding, DeepSeekNcclP2pDriver& nccl,
    CompletionEventDriver& event_driver) {
  if (state_ != DeepSeekBoundaryExecutorState::kReady ||
      transaction_->state() != DeepSeekPipelineTransactionState::kCommitted) {
    return Status::FailedPrecondition("DeepSeek boundary executor is not ready");
  }
  auto status = boundary_->bind(binding);
  if (!status.ok()) return poison(status);
  status = sequencer_->issue(*operation_, nccl);
  if (!status.ok()) return poison(status);
  if (operation_->state() == DeepSeekNcclP2pState::kGroupEndPending) {
    state_ = DeepSeekBoundaryExecutorState::kIssuePending;
    return Status::Ok();
  }
  return record_event(event_driver);
}

Status DeepSeekPipelineBoundaryExecutor::poll_issue(
    DeepSeekNcclP2pDriver& nccl, CompletionEventDriver& event_driver) {
  if (state_ != DeepSeekBoundaryExecutorState::kIssuePending) {
    return Status::FailedPrecondition("DeepSeek boundary issue is not pending");
  }
  const auto status = operation_->poll_issue(nccl);
  if (!status.ok()) return poison(status);
  if (operation_->state() == DeepSeekNcclP2pState::kGroupEndPending) {
    return Status::Unavailable("DeepSeek boundary issue remains pending");
  }
  return record_event(event_driver);
}

Status DeepSeekPipelineBoundaryExecutor::poll_completion(
    DeepSeekNcclP2pDriver& nccl, CompletionEventDriver& event_driver,
    CompletionEvidenceProvider& evidence) {
  if (state_ != DeepSeekBoundaryExecutorState::kDeviceInFlight) {
    return Status::FailedPrecondition("DeepSeek boundary is not in flight");
  }
  const auto status = operation_->poll_completion(
      nccl, *completion_event_, event_driver, *completion_frontier_, evidence);
  if (!status.ok()) {
    if (status.code() == StatusCode::kUnavailable &&
        operation_->state() != DeepSeekNcclP2pState::kPoisoned) {
      return status;
    }
    return poison(status);
  }
  const auto closed = sequencer_->close(*operation_);
  if (!closed.ok()) return poison(closed);
  state_ = DeepSeekBoundaryExecutorState::kCompleteVerified;
  return Status::Ok();
}

Status DeepSeekPipelineBoundaryExecutor::expire(std::uint64_t now_ns) {
  if (state_ == DeepSeekBoundaryExecutorState::kCompleteVerified) {
    return Status::FailedPrecondition(
        "completed DeepSeek boundary cannot expire");
  }
  if (state_ == DeepSeekBoundaryExecutorState::kPoisoned) {
    return Status::FailedPrecondition("DeepSeek boundary is poisoned");
  }
  const auto status = completion_frontier_->expire(now_ns);
  if (status.code() == StatusCode::kUnavailable) return status;
  return poison(status);
}

}  // namespace pih
