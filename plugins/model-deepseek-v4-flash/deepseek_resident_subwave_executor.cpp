#include "pih/model/deepseek_resident_subwave_executor.h"

namespace pih {

Result<DeepSeekResidentSubwaveExecutor>
DeepSeekResidentSubwaveExecutor::Create(
    std::uint32_t layer, const DeepSeekExpertSubwavePlan& plan,
    const DeepSeekResidentExpertBindings& bindings) {
  if (layer >= 43 || plan.token_count() == 0 ||
      bindings.generation() == 0 || !bindings.contains(layer)) {
    return Status::InvalidArgument(
        "DeepSeek resident subwave inputs are invalid");
  }
  DeepSeekResidentSubwaveExecutor result;
  result.layer_ = static_cast<std::uint16_t>(layer);
  result.plan_ = &plan;
  result.bindings_ = &bindings;
  return result;
}

Status DeepSeekResidentSubwaveExecutor::poison(Status status) {
  state_ = DeepSeekExpertSubwaveExecutorState::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek resident subwave poisoned")
                     : status;
}

Status DeepSeekResidentSubwaveExecutor::advance(
    DeepSeekExpertKernelDriver& kernel) {
  if (state_ == DeepSeekExpertSubwaveExecutorState::kComplete ||
      state_ == DeepSeekExpertSubwaveExecutorState::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek resident subwave is terminal");
  }
  if (state_ == DeepSeekExpertSubwaveExecutorState::kCompute) {
    auto status = kernel.poll();
    if (!status.ok()) return poison(status.status());
    if (*status == DeepSeekExpertAsyncStatus::kError) {
      return poison(Status::Internal("DeepSeek resident expert kernel failed"));
    }
    if (*status == DeepSeekExpertAsyncStatus::kInProgress) return Status::Ok();
    if (*status != DeepSeekExpertAsyncStatus::kSuccess) {
      return poison(
          Status::Internal("DeepSeek resident expert kernel returned invalid state"));
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
      {layer_, expert}, bindings_->generation(),
      bindings_->expert(layer_, expert), plan_->routes().data() + begin,
      count);
  if (!status.ok()) return poison(status);
  state_ = DeepSeekExpertSubwaveExecutorState::kCompute;
  return Status::Ok();
}

}  // namespace pih
