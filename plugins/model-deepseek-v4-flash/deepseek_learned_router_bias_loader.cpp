#include "pih/model/deepseek_learned_router_bias_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace pih {

Result<std::vector<DeepSeekLearnedRouterLayerBias>>
DeepSeekLearnedRouterBiasLoader::Load(
    DeepSeekStageRange owned_layers,
    const DeepSeekWeightByteSource& source) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer >= 43) {
    return Status::InvalidArgument(
        "DeepSeek learned router bias stage is invalid");
  }
  std::vector<DeepSeekLearnedRouterLayerBias> result;
  const auto first = std::max(owned_layers.first_layer, 3U);
  if (first > owned_layers.last_layer) return result;
  result.reserve(owned_layers.last_layer - first + 1U);
  for (std::uint32_t layer = first; layer <= owned_layers.last_layer;
       ++layer) {
    const auto name = "layers." + std::to_string(layer) +
                      ".ffn.gate.bias";
    auto bytes = source.resolve_weight_bytes(name);
    if (!bytes.ok()) return bytes.status();
    DeepSeekLearnedRouterLayerBias value;
    value.layer = layer;
    if (bytes->size_bytes() != sizeof(value.bias)) {
      return Status::InvalidArgument(
          "DeepSeek learned router bias byte extent is invalid");
    }
    std::memcpy(value.bias.data(), bytes->data(), sizeof(value.bias));
    if (std::ranges::any_of(value.bias,
                            [](float item) { return !std::isfinite(item); })) {
      return Status::InvalidArgument(
          "DeepSeek learned router bias contains non-finite data");
    }
    result.push_back(std::move(value));
  }
  return result;
}

}  // namespace pih
