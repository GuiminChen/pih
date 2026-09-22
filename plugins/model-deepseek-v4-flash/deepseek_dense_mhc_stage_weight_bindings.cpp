#include "pih/model/deepseek_dense_mhc_stage_weight_bindings.h"

namespace pih {

Result<DeepSeekDenseMhcStageWeightBindings>
DeepSeekDenseMhcStageWeightBindings::Resolve(
    DeepSeekStageRange owned_layers, const Resolver& resolver) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || !resolver) {
    return Status::InvalidArgument(
        "DeepSeek dense mHC stage weight coverage is invalid");
  }
  DeepSeekDenseMhcStageWeightBindings result;
  result.owned_layers_ = owned_layers;
  result.layers_.reserve(
      owned_layers.last_layer - owned_layers.first_layer + 1);
  for (std::uint32_t layer = owned_layers.first_layer;
       layer <= owned_layers.last_layer; ++layer) {
    auto attention = DeepSeekAttentionWeightBindings::Resolve(
        layer, resolver);
    if (!attention.ok()) return attention.status();
    auto mhc = DeepSeekMhcWeightBindings::Resolve(layer, resolver);
    if (!mhc.ok()) return mhc.status();
    if (attention->generation != mhc->generation ||
        (result.generation_ != 0 &&
         result.generation_ != attention->generation)) {
      return Status::FailedPrecondition(
          "DeepSeek dense mHC stage weight generations differ");
    }
    result.generation_ = attention->generation;
    result.layers_.push_back(
        {layer, std::move(*attention), std::move(*mhc)});
  }
  return result;
}

Result<DeepSeekDenseMhcStageWeightBindings>
DeepSeekDenseMhcStageWeightBindings::Resolve(
    DeepSeekStageRange owned_layers,
    const DeepSeekResidentWeightArena& arena) {
  auto result = Resolve(owned_layers, [&arena](std::string_view name) {
    return arena.tensor(name);
  });
  if (!result.ok()) return result.status();
  if (result->generation() != arena.generation()) {
    return Status::FailedPrecondition(
        "DeepSeek dense mHC stage weights are foreign to resident arena");
  }
  return result;
}

}  // namespace pih
