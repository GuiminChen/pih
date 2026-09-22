#include "pih/model/deepseek_pipeline_coordinator.h"

namespace pih {

Result<DeepSeekPipelineCoordinator> DeepSeekPipelineCoordinator::Create(
    DeepSeekPipelineTransaction& transaction, std::uint32_t world_size) {
  if (transaction.state() != DeepSeekPipelineTransactionState::kPrepared ||
      world_size < 1 || world_size > 4) {
    return Status::InvalidArgument(
        "DeepSeek coordinator requires a prepared PP1-PP4 transaction");
  }
  return DeepSeekPipelineCoordinator(transaction, world_size);
}

Status DeepSeekPipelineCoordinator::validate_rank(std::uint32_t rank) const {
  return rank < world_size_
             ? Status::Ok()
             : Status::InvalidArgument("DeepSeek pipeline rank is out of range");
}

Status DeepSeekPipelineCoordinator::poison(Status reason) {
  state_ = DeepSeekPipelineCoordinatorState::kPoisoned;
  suppress_output_ = true;
  return reason.ok() ? Status::Internal("DeepSeek pipeline coordinator poisoned")
                     : reason;
}

Status DeepSeekPipelineCoordinator::stage_ready(std::uint32_t rank) {
  const auto rank_status = validate_rank(rank);
  if (!rank_status.ok()) return rank_status;
  if (state_ != DeepSeekPipelineCoordinatorState::kPreparing &&
      state_ != DeepSeekPipelineCoordinatorState::kReadyToCommit) {
    return Status::FailedPrecondition(
        "DeepSeek stage READY is outside PREPARE");
  }
  if (ready_[rank]) {
    return Status::FailedPrecondition("DeepSeek stage READY is duplicated");
  }
  ready_[rank] = true;
  ++ready_count_;
  if (ready_count_ == world_size_) {
    state_ = DeepSeekPipelineCoordinatorState::kReadyToCommit;
  }
  return Status::Ok();
}

Status DeepSeekPipelineCoordinator::validate_stage_reject(
    std::uint32_t rank) const {
  const auto rank_status = validate_rank(rank);
  if (!rank_status.ok()) return rank_status;
  if (state_ != DeepSeekPipelineCoordinatorState::kPreparing &&
      state_ != DeepSeekPipelineCoordinatorState::kReadyToCommit) {
    return Status::FailedPrecondition(
        "DeepSeek stage rejection is outside PREPARE");
  }
  return transaction_->validate_abort_prepare();
}

Status DeepSeekPipelineCoordinator::stage_reject(std::uint32_t rank,
                                                  Status reason) {
  const auto validation = validate_stage_reject(rank);
  if (!validation.ok()) return validation;
  const auto abort_status = transaction_->abort_prepare();
  if (!abort_status.ok()) return poison(abort_status);
  state_ = DeepSeekPipelineCoordinatorState::kAborted;
  suppress_output_ = true;
  return reason.ok() ? Status::Unavailable("DeepSeek stage rejected PREPARE")
                     : reason;
}

Status DeepSeekPipelineCoordinator::validate_commit() const {
  if (state_ != DeepSeekPipelineCoordinatorState::kReadyToCommit ||
      ready_count_ != world_size_) {
    return Status::FailedPrecondition(
        "DeepSeek pipeline cannot COMMIT before all ranks are READY");
  }
  return transaction_->validate_commit();
}

Status DeepSeekPipelineCoordinator::commit() {
  const auto validation = validate_commit();
  if (!validation.ok()) return validation;
  const auto status = transaction_->commit();
  if (!status.ok()) return poison(status);
  state_ = DeepSeekPipelineCoordinatorState::kCommitted;
  return Status::Ok();
}

Status DeepSeekPipelineCoordinator::cancel_after_commit() {
  if (state_ != DeepSeekPipelineCoordinatorState::kCommitted &&
      state_ != DeepSeekPipelineCoordinatorState::kDraining) {
    return Status::FailedPrecondition(
        "DeepSeek committed plan is not available to drain");
  }
  suppress_output_ = true;
  state_ = DeepSeekPipelineCoordinatorState::kDraining;
  return Status::Ok();
}

Status DeepSeekPipelineCoordinator::validate_stage_complete(
    std::uint32_t rank) const {
  const auto rank_status = validate_rank(rank);
  if (!rank_status.ok()) return rank_status;
  if (state_ != DeepSeekPipelineCoordinatorState::kCommitted &&
      state_ != DeepSeekPipelineCoordinatorState::kDraining) {
    return Status::FailedPrecondition(
        "DeepSeek stage completion is outside committed execution");
  }
  if (complete_[rank]) {
    return Status::FailedPrecondition(
        "DeepSeek stage completion is duplicated");
  }
  if (complete_count_ + 1 == world_size_) {
    return transaction_->validate_complete();
  }
  return Status::Ok();
}

Status DeepSeekPipelineCoordinator::stage_complete(std::uint32_t rank) {
  const auto validation = validate_stage_complete(rank);
  if (!validation.ok()) return validation;
  complete_[rank] = true;
  ++complete_count_;
  if (complete_count_ != world_size_) return Status::Ok();
  const auto status = transaction_->complete();
  if (!status.ok()) return poison(status);
  state_ = DeepSeekPipelineCoordinatorState::kComplete;
  return Status::Ok();
}

Status DeepSeekPipelineCoordinator::validate_stage_failed(
    std::uint32_t rank) const {
  const auto rank_status = validate_rank(rank);
  if (!rank_status.ok()) return rank_status;
  if (state_ != DeepSeekPipelineCoordinatorState::kCommitted &&
      state_ != DeepSeekPipelineCoordinatorState::kDraining) {
    return Status::FailedPrecondition(
        "DeepSeek stage failure is outside committed execution");
  }
  return Status::Ok();
}

Status DeepSeekPipelineCoordinator::stage_failed(std::uint32_t rank,
                                                  Status reason) {
  const auto validation = validate_stage_failed(rank);
  if (!validation.ok()) return validation;
  return poison(reason);
}

}  // namespace pih
