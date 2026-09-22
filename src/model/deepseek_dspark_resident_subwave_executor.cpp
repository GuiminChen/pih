#include "pih/model/deepseek_dspark_resident_subwave_executor.h"

namespace pih {

Result<DeepSeekDsparkResidentSubwaveExecutor>
DeepSeekDsparkResidentSubwaveExecutor::Create(
    DeepSeekDsparkStageId stage,
    const DeepSeekExpertSubwavePlan& plan,
    const DeepSeekDsparkResidentExpertBindings& bindings) {
  if (!is_valid_deepseek_dspark_stage(stage) || plan.token_count() == 0 ||
      bindings.generation() == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark resident subwave inputs are invalid");
  }
  DeepSeekDsparkResidentSubwaveExecutor result;
  result.stage_ = stage;
  result.plan_ = &plan;
  result.bindings_ = &bindings;
  return result;
}

Status DeepSeekDsparkResidentSubwaveExecutor::poison(Status status) {
  state_ = DeepSeekExpertSubwaveExecutorState::kPoisoned;
  return status.ok()
             ? Status::Internal("DeepSeek DSpark resident subwave poisoned")
             : status;
}

Status DeepSeekDsparkResidentSubwaveExecutor::advance(
    DeepSeekDsparkExpertKernelDriver& kernel) {
  if (state_ == DeepSeekExpertSubwaveExecutorState::kComplete ||
      state_ == DeepSeekExpertSubwaveExecutorState::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark resident subwave is terminal");
  }
  if (state_ == DeepSeekExpertSubwaveExecutorState::kCompute) {
    auto status = kernel.poll();
    if (!status.ok()) return poison(status.status());
    if (*status == DeepSeekExpertAsyncStatus::kError) {
      return poison(Status::Internal(
          "DeepSeek DSpark resident expert kernel failed"));
    }
    if (*status == DeepSeekExpertAsyncStatus::kInProgress) return Status::Ok();
    if (*status != DeepSeekExpertAsyncStatus::kSuccess) {
      return poison(Status::Internal(
          "DeepSeek DSpark resident expert kernel returned invalid state"));
    }
    ++completed_experts_;
    state_ = DeepSeekExpertSubwaveExecutorState::kReady;
  }
  const auto& offsets = plan_->expert_offsets();
  while (next_expert_ < DeepSeekExpertSubwavePlan::kExpertCount &&
         offsets[next_expert_] == offsets[next_expert_ + 1]) {
    ++next_expert_;
  }
  if (next_expert_ == DeepSeekExpertSubwavePlan::kExpertCount) {
    state_ = DeepSeekExpertSubwaveExecutorState::kComplete;
    return Status::Ok();
  }
  const auto expert = next_expert_++;
  const auto begin = offsets[expert];
  const auto count = offsets[expert + 1] - begin;
  const auto status = kernel.launch_resident(
      stage_, expert, bindings_->generation(), bindings_->expert(stage_, expert),
      plan_->routes().data() + begin, count);
  if (!status.ok()) return poison(status);
  state_ = DeepSeekExpertSubwaveExecutorState::kCompute;
  return Status::Ok();
}

}  // namespace pih
