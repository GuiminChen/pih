#include "pih/model/deepseek_learned_router_coordinator.h"

#include <cmath>

namespace pih {
Result<DeepSeekLearnedRouterCoordinator>
DeepSeekLearnedRouterCoordinator::Create(
    std::uint32_t maximum_tokens, std::array<float, kExpertCount> bias,
    DeepSeekRouteScratchArena& scratch, DeepSeekExpertPlanStore& store,
    DeepSeekLearnedRouterOperations& operations) {
  if (maximum_tokens == 0 || maximum_tokens > 4096 ||
      scratch.maximum_token_count() < maximum_tokens) {
    return Status::InvalidArgument(
        "DeepSeek learned router capacity is invalid");
  }
  for (float value : bias) if (!std::isfinite(value))
    return Status::InvalidArgument("DeepSeek learned router bias is invalid");
  DeepSeekLearnedRouterCoordinator result;
  result.maximum_tokens_ = maximum_tokens;
  result.bias_ = bias;
  result.scratch_ = &scratch;
  result.store_ = &store;
  result.operations_ = &operations;
  return result;
}
Status DeepSeekLearnedRouterCoordinator::poison(Status status) {
  poisoned_ = true; inflight_ = false;
  return status.ok() ? Status::Internal("DeepSeek learned router poisoned")
                     : status;
}
Status DeepSeekLearnedRouterCoordinator::launch(
    const DeepSeekLearnedRouterSubmission& value) {
  const auto scores = static_cast<std::size_t>(value.token_count) * kExpertCount;
  if (inflight_ || poisoned_ || value.layer < 3 || value.layer > 42 ||
      value.token_count == 0 || value.token_count > maximum_tokens_ ||
      value.input_bf16 == 0 || value.weight_bf16 == 0 ||
      value.scores_f32 == 0 ||
      value.error_flag_u32 == 0 || value.stream == 0 ||
      value.completion_event == 0 || value.host_scores.size() != scores ||
      value.host_error_flag == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek learned router submission is invalid");
  }
  *value.host_error_flag = 0;
  auto run = [this](Status status) {
    return status.ok() ? status : poison(status);
  };
  auto status = run(operations_->zero_u32_async(value.error_flag_u32,
                                                 value.stream));
  if (!status.ok()) return status;
  status = run(operations_->gemm(
      {value.input_bf16, value.weight_bf16, value.scores_f32,
       value.error_flag_u32, value.stream, value.token_count, kExpertCount,
       kHiddenSize}));
  if (!status.ok()) return status;
  status = run(operations_->copy_d2h_async(
      value.host_scores.data(), value.scores_f32,
      scores * sizeof(float), value.stream));
  if (!status.ok()) return status;
  status = run(operations_->copy_d2h_async(
      value.host_error_flag, value.error_flag_u32, sizeof(std::uint32_t),
      value.stream));
  if (!status.ok()) return status;
  status = run(operations_->record_event(value.completion_event,
                                          value.stream));
  if (!status.ok()) return status;
  active_ = value; inflight_ = true;
  return Status::Ok();
}
Result<DeepSeekStageComputeStatus> DeepSeekLearnedRouterCoordinator::poll() {
  if (!inflight_ || poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek learned router is not pollable");
  }
  auto event = operations_->query_event(active_.completion_event);
  if (!event.ok()) return poison(event.status());
  if (*event == DeepSeekExpertAsyncStatus::kInProgress)
    return DeepSeekStageComputeStatus::kInProgress;
  if (*event == DeepSeekExpertAsyncStatus::kError ||
      *active_.host_error_flag != 0) {
    (void)poison(Status::Internal("DeepSeek learned router GPU failed"));
    return DeepSeekStageComputeStatus::kError;
  }
  if (*event != DeepSeekExpertAsyncStatus::kSuccess) {
    return poison(Status::Internal(
        "DeepSeek learned router received invalid event state"));
  }
  auto status = DeepSeekLearnedRouter::RouteInto(
      active_.layer, active_.token_count, active_.host_scores, bias_,
      *scratch_, *store_);
  if (!status.ok()) return poison(status);
  inflight_ = false;
  return DeepSeekStageComputeStatus::kSuccess;
}
}  // namespace pih
