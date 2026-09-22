#pragma once

#include <array>
#include <memory>
#include <vector>

#include "pih/model/deepseek_learned_router_host_staging.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

struct DeepSeekLearnedRouterLayerStaging final {
  std::uint32_t layer = 0;
  std::shared_ptr<DeepSeekLearnedRouterHostStaging> staging;
};

class DeepSeekLearnedRouterStagingPool final {
 public:
  static Result<DeepSeekLearnedRouterStagingPool> Allocate(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
      RegisteredPinnedAllocator& allocator);

  Result<std::vector<DeepSeekLearnedRouterLayerStaging>> acquire();
  [[nodiscard]] std::uint32_t layer_count() const noexcept {
    return layer_count_;
  }

 private:
  DeepSeekStageRange owned_layers_;
  std::array<std::shared_ptr<DeepSeekLearnedRouterHostStaging>, 43> staging_;
  std::uint32_t layer_count_ = 0;
};

}  // namespace pih
