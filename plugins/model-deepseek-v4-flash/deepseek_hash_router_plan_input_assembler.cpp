#include "pih/model/deepseek_hash_router_plan_input_assembler.h"

namespace pih {

Result<std::vector<DeepSeekHashRouterLayerPlanWork>>
DeepSeekHashRouterPlanInputAssembler::Assemble(
    std::uint32_t token_count,
    std::span<const DeepSeekLearnedRouterLayerInput> layer_inputs,
    const DeepSeekHashRouterWeightBindings& weights,
    DeepSeekLearnedRouterDeviceView device,
    DeepSeekHashRouterStagingPool& staging_pool,
    std::uintptr_t stream, std::uintptr_t completion_event,
    std::uint32_t maximum_tokens) {
  if (token_count == 0 || token_count > maximum_tokens ||
      maximum_tokens == 0 || maximum_tokens > 4096 ||
      device.scores_f32 == 0 || device.error_flag_u32 == 0 ||
      stream == 0 || completion_event == 0 ||
      (weights.size() != 0 && weights.generation() == 0) ||
      weights.size() != staging_pool.layer_count() ||
      layer_inputs.size() != weights.size()) {
    return Status::InvalidArgument(
        "DeepSeek hash router plan input identity is invalid");
  }
  for (std::size_t index = 0; index < weights.size(); ++index) {
    const auto& binding = weights.at(index);
    const auto& input = layer_inputs[index];
    if (binding.layer != input.layer) {
      return Status::FailedPrecondition(
          "DeepSeek hash router plan layer ownership is inconsistent");
    }
    auto status = validate_deepseek_router_bf16_gemm_launch(
        {input.input_bf16, binding.weight_bf16, device.scores_f32,
         device.error_flag_u32, stream, token_count,
         DeepSeekLearnedRouterCoordinator::kExpertCount,
         DeepSeekLearnedRouterCoordinator::kHiddenSize});
    if (!status.ok()) return status;
  }
  auto staging = staging_pool.acquire();
  if (!staging.ok()) return staging.status();
  if (staging->size() != weights.size()) {
    return Status::Internal(
        "DeepSeek hash router staging and weights disagree");
  }
  std::vector<DeepSeekHashRouterLayerPlanWork> result;
  result.reserve(weights.size());
  for (std::size_t index = 0; index < weights.size(); ++index) {
    const auto& binding = weights.at(index);
    const auto& input = layer_inputs[index];
    auto& layer_staging = staging->at(index);
    if (binding.layer != input.layer ||
        binding.layer != layer_staging.layer || input.input_bf16 == 0 ||
        binding.weight_bf16 == 0 || layer_staging.staging == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek hash router plan layer ownership is inconsistent");
    }
    auto host_scores = layer_staging.staging->scores(token_count);
    if (!host_scores.ok()) return host_scores.status();
    auto* host_error = layer_staging.staging->error_flag();
    if (host_error == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek hash router host error flag is unavailable");
    }
    DeepSeekLearnedRouterSubmission submission{
        binding.layer, token_count, input.input_bf16,
        binding.weight_bf16, device.scores_f32, device.error_flag_u32,
        stream, completion_event, *host_scores, host_error};
    result.push_back({submission, std::move(layer_staging.staging)});
  }
  return result;
}

}  // namespace pih
