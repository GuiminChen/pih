#include "pih/model/deepseek_dspark_mtp_block_executor.h"

namespace pih {
Result<DeepSeekDsparkMtpBlockExecutor>
DeepSeekDsparkMtpBlockExecutor::Create(
    DeepSeekDsparkMtpOperatorBackend& backend) {
  DeepSeekDsparkMtpBlockExecutor value;
  value.backend_ = &backend;
  return value;
}
Status DeepSeekDsparkMtpBlockExecutor::poison(Status status) {
  state_ = State::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek DSpark MTP block poisoned")
                     : status;
}
Status DeepSeekDsparkMtpBlockExecutor::launch(
    DeepSeekDsparkStageId stage_id,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (state_ != State::kIdle || !is_valid_deepseek_dspark_stage(stage_id) ||
      plan.engine_epoch == 0 || plan.plan_sequence == 0 ||
      plan.token_count == 0 || plan.sequence_count == 0 ||
      (plan.phase != DeepSeekPlanPhase::kPrefill &&
       plan.phase != DeepSeekPlanPhase::kDecode)) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark MTP block is not launchable");
  }
  stage_id_ = stage_id;
  plan_ = plan;
  auto status = backend_->launch(
      {stage_id_, DeepSeekDsparkMtpOperatorKind::kAttention}, plan_);
  if (!status.ok()) return poison(status);
  state_ = State::kAttention;
  return Status::Ok();
}
Result<DeepSeekStageComputeStatus> DeepSeekDsparkMtpBlockExecutor::poll() {
  if (state_ == State::kIdle || state_ == State::kPoisoned)
    return Status::FailedPrecondition(
        "DeepSeek DSpark MTP block is not pollable");
  auto result = backend_->poll();
  if (!result.ok()) return poison(result.status());
  if (*result == DeepSeekStageComputeStatus::kInProgress) return *result;
  if (*result == DeepSeekStageComputeStatus::kError) {
    (void)poison(Status::Internal("DeepSeek DSpark MTP operator failed"));
    return DeepSeekStageComputeStatus::kError;
  }
  if (state_ == State::kAttention) {
    if (plan_.phase == DeepSeekPlanPhase::kPrefill) {
      state_ = State::kIdle;
      return DeepSeekStageComputeStatus::kSuccess;
    }
    auto status = backend_->launch(
        {stage_id_, DeepSeekDsparkMtpOperatorKind::kMoe}, plan_);
    if (!status.ok()) return poison(status);
    state_ = State::kMoe;
    return DeepSeekStageComputeStatus::kInProgress;
  }
  state_ = State::kIdle;
  return DeepSeekStageComputeStatus::kSuccess;
}

Status DeepSeekDsparkMtpBlockExecutor::cancel() {
  if (state_ != State::kAttention && state_ != State::kMoe) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark MTP block has no cancellable operation");
  }
  const auto status = backend_->cancel();
  state_ = State::kPoisoned;
  return status;
}

}  // namespace pih
