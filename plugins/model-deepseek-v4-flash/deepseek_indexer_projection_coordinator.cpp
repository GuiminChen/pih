#include "pih/model/deepseek_indexer_projection_coordinator.h"

namespace pih {

Result<DeepSeekIndexerProjectionCoordinator>
DeepSeekIndexerProjectionCoordinator::Create(
    DeepSeekIndexerProjectionOperations& operations,
    std::uint32_t* host_error_flag) {
  if (host_error_flag == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek indexer projection host error is null");
  }
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekIndexerProjectionCoordinator result;
  result.operations_ = &operations;
  result.host_error_flag_ = host_error_flag;
  return result;
}

Status DeepSeekIndexerProjectionCoordinator::run(Status status) {
  if (!status.ok()) poisoned_ = true;
  return status;
}

Status DeepSeekIndexerProjectionCoordinator::launch(
    const DeepSeekIndexerProjectionLaunch& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ || transaction.state() !=
                       DeepSeekAttentionSequenceTransactionState::kPreparing ||
      transaction.stream() != submission.stream) {
    return Status::FailedPrecondition(
        "DeepSeek indexer projection coordinator is not launchable");
  }
  auto status = validate_deepseek_indexer_projection_launch(submission);
  if (!status.ok()) return status;
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, submission.error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  if (*claimed) {
    *host_error_flag_ = 0;
    status = run(operations_->zero_u32_async(
        submission.error_flag_u32, submission.stream));
    if (!status.ok()) return status;
  }
  status = run(operations_->project(submission));
  if (!status.ok()) return status;
  return run(operations_->copy_error_d2h_async(
      host_error_flag_, submission.error_flag_u32, submission.stream));
}

}  // namespace pih
