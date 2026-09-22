#include "pih/model/deepseek_hash_router_staging_pool.h"

#include <algorithm>

namespace pih {

Result<DeepSeekHashRouterStagingPool>
DeepSeekHashRouterStagingPool::Allocate(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
    RegisteredPinnedAllocator& allocator) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || maximum_tokens == 0 ||
      maximum_tokens > 4096) {
    return Status::InvalidArgument(
        "DeepSeek hash router staging pool identity is invalid");
  }
  DeepSeekHashRouterStagingPool result;
  result.owned_layers_ = owned_layers;
  const auto last = std::min(owned_layers.last_layer, 2U);
  if (owned_layers.first_layer > last) return result;
  for (auto layer = owned_layers.first_layer; layer <= last; ++layer) {
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

Result<std::vector<DeepSeekHashRouterLayerStaging>>
DeepSeekHashRouterStagingPool::acquire() {
  std::vector<DeepSeekHashRouterLayerStaging> result;
  result.reserve(layer_count_);
  const auto last = std::min(owned_layers_.last_layer, 2U);
  if (owned_layers_.first_layer > last) return result;
  for (auto layer = owned_layers_.first_layer; layer <= last; ++layer) {
    if (staging_[layer] == nullptr || staging_[layer].use_count() != 1) {
      return Status::ResourceExhausted(
          "DeepSeek hash router staging is leased by another plan");
    }
  }
  for (auto layer = owned_layers_.first_layer; layer <= last; ++layer) {
    result.push_back({layer, staging_[layer]});
  }
  return result;
}

}  // namespace pih
