#pragma once

#include <array>
#include <optional>
#include <span>

#include "pih/model/deepseek_routed_stage_backend.h"

namespace pih {

struct DeepSeekLayerExpertRoutes final {
  std::uint32_t layer = 0;
  std::span<const DeepSeekExpertRoute> routes;
};

// Double-buffered, allocation-stable provider for one rank's routed layers.
// A failed bind never publishes a partial layer set or changes the active plan.
class DeepSeekBoundExpertPlanProvider final : public DeepSeekExpertPlanStore {
 public:
  static Result<DeepSeekBoundExpertPlanProvider> Create(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_token_count);

  Status bind(const DeepSeekPipelinePlanDescriptor& descriptor,
              std::span<const DeepSeekLayerExpertRoutes> layers);
  Status begin(const DeepSeekPipelinePlanDescriptor& descriptor);
  Status publish_routes(std::uint32_t layer, std::uint32_t token_count,
                        std::span<const DeepSeekExpertRoute> routes) override;
  void clear() noexcept { active_descriptor_.reset(); }

  Result<const DeepSeekExpertSubwavePlan*> resolve(
      std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& descriptor) override;

 private:
  DeepSeekStageRange owned_layers_;
  std::uint32_t maximum_token_count_ = 0;
  std::array<std::vector<DeepSeekExpertSubwavePlan>, 2> banks_;
  std::size_t active_bank_ = 0;
  std::optional<DeepSeekPipelinePlanDescriptor> active_descriptor_;
  std::array<bool, 43> published_{};
};

}  // namespace pih
