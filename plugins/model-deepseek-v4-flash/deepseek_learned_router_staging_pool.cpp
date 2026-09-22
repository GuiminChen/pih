#include "pih/model/deepseek_learned_router_staging_pool.h"

#include <algorithm>

namespace pih {

Result<DeepSeekLearnedRouterStagingPool>
DeepSeekLearnedRouterStagingPool::Allocate(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
    RegisteredPinnedAllocator& allocator) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || maximum_tokens == 0 ||
      maximum_tokens > 4096) {
    return Status::InvalidArgument(
        "DeepSeek learned router staging pool identity is invalid");
  }
  DeepSeekLearnedRouterStagingPool result;
  result.owned_layers_ = owned_layers;
  const auto first = std::max(owned_layers.first_layer, 3U);
  if (first > owned_layers.last_layer) return result;
  for (std::uint32_t layer = first; layer <= owned_layers.last_layer;
       ++layer) {
    auto staging = DeepSeekLearnedRouterHostStaging::Allocate(
        maximum_tokens, allocator);
    if (!staging.ok()) return staging.status();
    result.staging_[layer] =
        std::make_shared<DeepSeekLearnedRouterHostStaging>(
            std::move(*staging));
    ++result.layer_count_;
  }
  return result;
}

Result<std::vector<DeepSeekLearnedRouterLayerStaging>>
DeepSeekLearnedRouterStagingPool::acquire() {
  std::vector<DeepSeekLearnedRouterLayerStaging> result;
  result.reserve(layer_count_);
  const auto first = std::max(owned_layers_.first_layer, 3U);
  if (first > owned_layers_.last_layer) return result;
  for (std::uint32_t layer = first; layer <= owned_layers_.last_layer;
       ++layer) {
    if (staging_[layer] == nullptr || staging_[layer].use_count() != 1) {
      return Status::ResourceExhausted(
          "DeepSeek learned router staging is leased by another plan");
    }
  }
  for (std::uint32_t layer = first; layer <= owned_layers_.last_layer;
       ++layer) {
    result.push_back({layer, staging_[layer]});
  }
  return result;
}

}  // namespace pih
