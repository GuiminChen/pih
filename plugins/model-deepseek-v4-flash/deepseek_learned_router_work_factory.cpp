#include "pih/model/deepseek_learned_router_work_factory.h"

#include <algorithm>

namespace pih {
namespace {

bool complete_submission(const DeepSeekLearnedRouterSubmission& value) {
  return value.completion_event != 0 &&
         validate_deepseek_router_bf16_gemm_launch(
             {value.input_bf16, value.weight_bf16, value.scores_f32,
              value.error_flag_u32, value.stream, value.token_count,
              DeepSeekLearnedRouterCoordinator::kExpertCount,
              DeepSeekLearnedRouterCoordinator::kHiddenSize})
             .ok();
}

}  // namespace

Result<DeepSeekLearnedRouterWorkFactory>
DeepSeekLearnedRouterWorkFactory::Create(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
    std::vector<DeepSeekLearnedRouterLayerBias> biases,
    DeepSeekExpertPlanStore& store,
    DeepSeekLearnedRouterOperations& operations) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || maximum_tokens == 0 ||
      maximum_tokens > 4096) {
    return Status::InvalidArgument(
        "DeepSeek learned router work factory identity is invalid");
  }
  const auto first = std::max(owned_layers.first_layer, 3U);
  const auto expected = first <= owned_layers.last_layer
                            ? owned_layers.last_layer - first + 1
                            : 0U;
  if (biases.size() != expected) {
    return Status::InvalidArgument(
        "DeepSeek learned router biases do not cover owned layers");
  }
  DeepSeekLearnedRouterWorkFactory result;
  result.owned_layers_ = owned_layers;
  result.maximum_tokens_ = maximum_tokens;
  result.owned_learned_layer_count_ = expected;
  std::array<bool, 43> seen{};
  for (auto& layer : biases) {
    if (layer.layer < first || layer.layer > owned_layers.last_layer ||
        seen[layer.layer]) {
      return Status::InvalidArgument(
          "DeepSeek learned router bias layer is foreign or duplicated");
    }
    seen[layer.layer] = true;
    auto scratch = DeepSeekRouteScratchArena::Create(maximum_tokens);
    if (!scratch.ok()) return scratch.status();
    result.scratch_[layer.layer] = std::move(*scratch);
    auto coordinator = DeepSeekLearnedRouterCoordinator::Create(
        maximum_tokens, layer.bias, *result.scratch_[layer.layer],
        store, operations);
    if (!coordinator.ok()) return coordinator.status();
    result.coordinators_[layer.layer] = std::move(*coordinator);
  }
  return result;
}

Status DeepSeekLearnedRouterWorkFactory::append_plan_work(
    std::uint32_t token_count,
    std::vector<DeepSeekLearnedRouterLayerPlanWork> layers,
    DeepSeekRankComputeWorkBuilder& builder) {
  if (token_count == 0 || token_count > maximum_tokens_ ||
      layers.size() != owned_learned_layer_count_) {
    return Status::InvalidArgument(
        "DeepSeek learned router plan capacity or layer count is invalid");
  }
  std::array<bool, 43> seen{};
  for (auto& layer : layers) {
    const auto id = layer.submission.layer;
    if (id >= coordinators_.size() || !coordinators_[id].has_value() ||
        seen[id] || layer.submission.token_count != token_count ||
        layer.host_staging == nullptr ||
        !complete_submission(layer.submission)) {
      return Status::InvalidArgument(
          "DeepSeek learned router plan layer is invalid or duplicated");
    }
    seen[id] = true;
    const auto status = builder.add_learned_router_with_host_staging(
        id, &*coordinators_[id], layer.submission,
        std::move(layer.host_staging));
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace pih
