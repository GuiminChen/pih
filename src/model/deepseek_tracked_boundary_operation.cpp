#include "pih/model/deepseek_tracked_boundary_operation.h"

#include <utility>

namespace pih {

Result<std::unique_ptr<DeepSeekTrackedBoundaryOperation>>
DeepSeekTrackedBoundaryOperation::Create(
    std::unique_ptr<DeepSeekPreparedBoundaryOperation> operation,
    DeepSeekBoundaryCreditTracker& tracker,
    DeepSeekBoundaryCreditHandle handle) {
  if (operation == nullptr ||
      operation->manifest().pipeline_plan_sequence !=
          handle.pipeline_plan_sequence ||
      operation->manifest().operation_ordinal != handle.operation_ordinal) {
    return Status::InvalidArgument(
        "DeepSeek tracked boundary operation identity is invalid");
  }
  const auto prepared = tracker.validate(
      handle, DeepSeekBoundaryCreditState::kPrepared);
  if (!prepared.ok()) return prepared;
  return std::unique_ptr<DeepSeekTrackedBoundaryOperation>(
      new DeepSeekTrackedBoundaryOperation(
          std::move(operation), tracker, handle));
}

DeepSeekTrackedBoundaryOperation::~DeepSeekTrackedBoundaryOperation() {
  if (tracker_ == nullptr) return;
  if (!committed_) {
    (void)tracker_->abort_prepare(handle_);
  } else if (complete_verified_) {
    (void)tracker_->release(handle_);
  } else {
    (void)tracker_->mark_suspect(handle_);
  }
}

Status DeepSeekTrackedBoundaryOperation::quarantine(Status cause) noexcept {
  if (committed_ && !complete_verified_) {
    (void)tracker_->mark_suspect(handle_);
  }
  return cause;
}

Status DeepSeekTrackedBoundaryOperation::submit(
    DeepSeekNcclP2pBindingTarget& binding, DeepSeekNcclP2pDriver& nccl,
    CompletionEventDriver& event_driver) {
  if (committed_) {
    return Status::FailedPrecondition(
        "DeepSeek tracked boundary was already submitted");
  }
  const auto committed = tracker_->commit(handle_);
  if (!committed.ok()) return committed;
  committed_ = true;
  const auto status = operation_->submit(binding, nccl, event_driver);
  return status.ok() ? status : quarantine(status);
}

Status DeepSeekTrackedBoundaryOperation::poll_issue(
    DeepSeekNcclP2pDriver& nccl, CompletionEventDriver& event_driver) {
  const auto status = operation_->poll_issue(nccl, event_driver);
  if (!status.ok() && status.code() != StatusCode::kUnavailable) {
    return quarantine(status);
  }
  return status;
}

Status DeepSeekTrackedBoundaryOperation::poll_completion(
    DeepSeekNcclP2pDriver& nccl, CompletionEventDriver& event_driver,
    CompletionEvidenceProvider& evidence) {
  const auto status = operation_->poll_completion(
      nccl, event_driver, evidence);
  if (!status.ok()) {
    return status.code() == StatusCode::kUnavailable
               ? status
               : quarantine(status);
  }
  if (operation_->state() ==
      DeepSeekBoundaryExecutorState::kCompleteVerified) {
    const auto completed = tracker_->complete_verified(handle_);
    if (!completed.ok()) return quarantine(completed);
    complete_verified_ = true;
  }
  return Status::Ok();
}

Status DeepSeekTrackedBoundaryOperation::expire(std::uint64_t now_ns) {
  const auto status = operation_->expire(now_ns);
  return status.code() == StatusCode::kUnavailable
             ? status
             : quarantine(status);
}

}  // namespace pih
