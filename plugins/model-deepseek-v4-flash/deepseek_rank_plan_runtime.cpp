#include "pih/model/deepseek_rank_plan_runtime.h"

#include <new>
#include <utility>

namespace pih {

DeepSeekRankPlanRuntime& DeepSeekRankPlanRuntime::operator=(
    DeepSeekRankPlanRuntime&& other) noexcept {
  if (this != &other) {
    // executor_ borrows transaction_; destroy the executor first by applying
    // the class's reverse destruction order before replacing either object.
    this->~DeepSeekRankPlanRuntime();
    ::new (static_cast<void*>(this)) DeepSeekRankPlanRuntime(std::move(other));
  }
  return *this;
}

Result<DeepSeekRankPlanRuntime>
DeepSeekRankPlanRuntime::CreateBorrowedCompute(
    DeepSeekPipelineTransaction transaction, DeepSeekStagePlan stage,
    DeepSeekRankPlanReservation reservation,
    DeepSeekStageComputeDriver& compute) {
  if (reservation.state() != DeepSeekRankPlanReservationState::kPrepared) {
    return Status::FailedPrecondition(
        "DeepSeek PP1 runtime requires a prepared resource reservation");
  }
  if (stage.rank != 0 || !stage.owns_embedding || !stage.owns_lm_head) {
    return Status::InvalidArgument(
        "DeepSeek PP1 runtime requires the complete single-rank stage");
  }
  return DeepSeekRankPlanRuntime(
      std::make_unique<DeepSeekPipelineTransaction>(std::move(transaction)),
      std::move(reservation), compute, stage, 1);
}

Status DeepSeekRankPlanRuntime::mark_ready() {
  if (state_ != DeepSeekRankPlanRuntimeState::kPrepared ||
      reservation_.state() != DeepSeekRankPlanReservationState::kPrepared) {
    return Status::FailedPrecondition(
        "DeepSeek rank runtime cannot report READY without prepared resources");
  }
  state_ = DeepSeekRankPlanRuntimeState::kReady;
  return Status::Ok();
}

Status DeepSeekRankPlanRuntime::prepare_commit() {
  if (state_ != DeepSeekRankPlanRuntimeState::kReady) {
    return Status::FailedPrecondition(
        "DeepSeek rank runtime requires READY before COMMIT");
  }
  if (executor_ != nullptr) return validate_commit();
  auto executor = DeepSeekPipelineStageExecutor::CreateStaged(
      *transaction_, stage_, world_size_);
  if (!executor.ok()) return executor.status();
  executor_ = std::make_unique<DeepSeekPipelineStageExecutor>(
      std::move(*executor));
  return validate_commit();
}

Status DeepSeekRankPlanRuntime::validate_commit() const {
  if (state_ != DeepSeekRankPlanRuntimeState::kReady ||
      executor_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank runtime COMMIT is not staged");
  }
  auto status = reservation_.validate_commit();
  if (!status.ok()) return status;
  return transaction_->validate_commit();
}

Status DeepSeekRankPlanRuntime::commit() {
  auto status = executor_ == nullptr ? prepare_commit() : validate_commit();
  if (!status.ok()) return status;
  status = reservation_.commit();
  if (!status.ok()) return poison(std::move(status));
  status = transaction_->commit();
  if (!status.ok()) return poison(std::move(status));
  state_ = DeepSeekRankPlanRuntimeState::kCommitted;
  return Status::Ok();
}

Status DeepSeekRankPlanRuntime::cancel() {
  if (state_ != DeepSeekRankPlanRuntimeState::kCommitted &&
      state_ != DeepSeekRankPlanRuntimeState::kExecuting) {
    return Status::FailedPrecondition(
        "DeepSeek rank runtime cannot cancel outside committed execution");
  }
  cancellation_requested_ = true;
  return Status::Ok();
}

Status DeepSeekRankPlanRuntime::advance_staged() {
  if (state_ != DeepSeekRankPlanRuntimeState::kCommitted &&
      state_ != DeepSeekRankPlanRuntimeState::kExecuting) {
    return Status::FailedPrecondition(
        "DeepSeek rank runtime cannot execute before COMMIT");
  }
  auto drivers = borrowed_drivers();
  auto status = executor_->advance(drivers);
  if (!status.ok() && status.code() != StatusCode::kUnavailable) {
    return poison(std::move(status));
  }
  if (executor_->state() == DeepSeekPipelineStageExecutorState::kComplete) {
    state_ = DeepSeekRankPlanRuntimeState::kCompletionReady;
    return Status::Ok();
  }
  state_ = DeepSeekRankPlanRuntimeState::kExecuting;
  return status;
}

Status DeepSeekRankPlanRuntime::validate_complete() const {
  if (state_ != DeepSeekRankPlanRuntimeState::kCompletionReady ||
      reservation_.state() != DeepSeekRankPlanReservationState::kCommitted) {
    return Status::FailedPrecondition(
        "DeepSeek rank runtime completion is not staged");
  }
  return transaction_->validate_complete();
}

Status DeepSeekRankPlanRuntime::complete() {
  const auto validation = validate_complete();
  if (!validation.ok()) return validation;
  const auto completed = transaction_->complete();
  if (!completed.ok()) return poison(completed);
  reservation_.release();
  state_ = DeepSeekRankPlanRuntimeState::kComplete;
  return Status::Ok();
}

Status DeepSeekRankPlanRuntime::advance() {
  auto status = advance_staged();
  if (!status.ok()) return status;
  return state_ == DeepSeekRankPlanRuntimeState::kCompletionReady
             ? complete()
             : status;
}

DeepSeekStageExecutionDrivers
DeepSeekRankPlanRuntime::borrowed_drivers() noexcept {
  return {compute_};
}

Status DeepSeekRankPlanRuntime::poison(Status cause) noexcept {
  state_ = DeepSeekRankPlanRuntimeState::kPoisoned;
  return cause;
}

}  // namespace pih
