#include "pih/model/deepseek_stage_compute_driver.h"

namespace pih {
namespace {

bool phase_runs_main_model(DeepSeekPlanPhase phase) {
  return phase == DeepSeekPlanPhase::kPrefill ||
         phase == DeepSeekPlanPhase::kDecode ||
         phase == DeepSeekPlanPhase::kVerify;
}

bool phase_runs_dspark(DeepSeekPlanPhase phase) {
  return phase == DeepSeekPlanPhase::kPrefill ||
         phase == DeepSeekPlanPhase::kDecode;
}

}  // namespace

Status DeepSeekModelStageComputeDriver::launch(
    const DeepSeekPipelinePlanDescriptor& plan,
    const DeepSeekStagePlan& stage) {
  if (launched_ || poisoned_ || plan.engine_epoch == 0 ||
      plan.plan_sequence == 0 || stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 ||
      (plan.phase == DeepSeekPlanPhase::kDrain && plan.token_count != 0) ||
      (plan.phase != DeepSeekPlanPhase::kDrain && plan.token_count == 0)) {
    return Status::FailedPrecondition(
        "DeepSeek stage compute launch contract is invalid");
  }
  plan_ = plan;
  commands_.clear();
  next_command_ = 0;
  launched_ = true;
  if (!phase_runs_main_model(plan.phase)) return Status::Ok();
  if (stage.owns_embedding) {
    commands_.push_back({DeepSeekStageOperatorKind::kEmbedding, 0});
  }
  for (std::uint32_t layer = stage.layers.first_layer;
       layer <= stage.layers.last_layer; ++layer) {
    commands_.push_back({DeepSeekStageOperatorKind::kAttention, layer});
    commands_.push_back({DeepSeekStageOperatorKind::kMoe, layer});
  }
  if (stage.owns_lm_head) {
    commands_.push_back({DeepSeekStageOperatorKind::kHead, 0});
  }
  if (stage.owns_dspark && phase_runs_dspark(plan.phase)) {
    commands_.push_back({DeepSeekStageOperatorKind::kDspark, 0});
  }
  const auto status = dispatch_next();
  if (!status.ok()) poisoned_ = true;
  return status;
}

Status DeepSeekModelStageComputeDriver::dispatch_next() {
  if (next_command_ >= commands_.size()) {
    command_inflight_ = false;
    return Status::Ok();
  }
  const auto status = backend_->launch(commands_[next_command_], plan_);
  if (!status.ok()) return status;
  command_inflight_ = true;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus> DeepSeekModelStageComputeDriver::poll() {
  if (!launched_ || poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek stage compute driver is not pollable");
  }
  if (!command_inflight_) {
    launched_ = false;
    return DeepSeekStageComputeStatus::kSuccess;
  }
  auto result = backend_->poll();
  if (!result.ok()) {
    poisoned_ = true;
    return result.status();
  }
  if (*result == DeepSeekStageComputeStatus::kError) {
    poisoned_ = true;
    return DeepSeekStageComputeStatus::kError;
  }
  if (*result == DeepSeekStageComputeStatus::kInProgress) return *result;
  ++next_command_;
  command_inflight_ = false;
  const auto status = dispatch_next();
  if (!status.ok()) {
    poisoned_ = true;
    return status;
  }
  if (command_inflight_) return DeepSeekStageComputeStatus::kInProgress;
  launched_ = false;
  return DeepSeekStageComputeStatus::kSuccess;
}

}  // namespace pih
