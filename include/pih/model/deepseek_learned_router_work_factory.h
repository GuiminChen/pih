#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "pih/model/deepseek_bound_expert_plan_provider.h"
#include "pih/model/deepseek_rank_compute_work_builder.h"

namespace pih {

struct DeepSeekLearnedRouterLayerBias final {
  std::uint32_t layer = 0;
  std::array<float, DeepSeekLearnedRouterCoordinator::kExpertCount> bias{};
};

struct DeepSeekLearnedRouterLayerPlanWork final {
  DeepSeekLearnedRouterSubmission submission;
  std::shared_ptr<DeepSeekLearnedRouterHostStaging> host_staging;
};

class DeepSeekLearnedRouterWorkFactory final {
 public:
  static Result<DeepSeekLearnedRouterWorkFactory> Create(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
      std::vector<DeepSeekLearnedRouterLayerBias> biases,
      DeepSeekExpertPlanStore& store,
      DeepSeekLearnedRouterOperations& operations);

  Status append_plan_work(
      std::uint32_t token_count,
      std::vector<DeepSeekLearnedRouterLayerPlanWork> layers,
      DeepSeekRankComputeWorkBuilder& builder);

  [[nodiscard]] std::uint32_t owned_learned_layer_count() const noexcept {
    return owned_learned_layer_count_;
  }
  [[nodiscard]] DeepSeekStageRange owned_layers() const noexcept {
    return owned_layers_;
  }

 private:
  DeepSeekStageRange owned_layers_;
  std::uint32_t maximum_tokens_ = 0;
  std::uint32_t owned_learned_layer_count_ = 0;
  std::array<std::optional<DeepSeekRouteScratchArena>, 43> scratch_;
  std::array<std::optional<DeepSeekLearnedRouterCoordinator>, 43>
      coordinators_;
};

}  // namespace pih
