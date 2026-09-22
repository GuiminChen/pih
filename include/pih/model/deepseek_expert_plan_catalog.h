#pragma once

#include <array>
#include <optional>

#include "pih/model/deepseek_routed_stage_backend.h"

namespace pih {

class DeepSeekExpertPlanCatalog final : public DeepSeekExpertPlanStore {
 public:
  static Result<DeepSeekExpertPlanCatalog> Create(
      DeepSeekPipelinePlanDescriptor descriptor,
      DeepSeekStageRange owned_layers);

  Status publish(std::uint32_t layer, DeepSeekExpertSubwavePlan plan);
  Status publish_routes(std::uint32_t layer, std::uint32_t token_count,
                        std::span<const DeepSeekExpertRoute> routes) override;
  Result<const DeepSeekExpertSubwavePlan*> resolve(
      std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& descriptor) override;
  [[nodiscard]] std::uint32_t published_layers() const noexcept {
    return published_layers_;
  }

 private:
  DeepSeekPipelinePlanDescriptor descriptor_{};
  DeepSeekStageRange owned_layers_{};
  std::array<std::optional<DeepSeekExpertSubwavePlan>, 43> plans_;
  std::array<bool, 43> published_{};
  std::uint32_t published_layers_ = 0;
};

}  // namespace pih
