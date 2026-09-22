#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "pih/model/deepseek_rank_compute_work_builder.h"
#include "pih/model/deepseek_hash_router_coordinator.h"

namespace pih {

struct DeepSeekHashRouterLayerTable final {
  std::uint32_t layer = 0;
  std::vector<std::uint16_t> token_to_experts;
};

struct DeepSeekHashRouterLayerScores final {
  std::uint32_t layer = 0;
  std::vector<float> raw_scores;
};

struct DeepSeekHashRouterLayerPlanWork final {
  DeepSeekLearnedRouterSubmission submission;
  std::shared_ptr<DeepSeekLearnedRouterHostStaging> host_staging;
};

class DeepSeekHashRouterWorkFactory final {
 public:
  static Result<DeepSeekHashRouterWorkFactory> Create(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
      std::uint32_t vocabulary_size,
      std::vector<DeepSeekHashRouterLayerTable> tables);
  static Result<DeepSeekHashRouterWorkFactory> CreateProjected(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
      std::uint32_t vocabulary_size,
      std::vector<DeepSeekHashRouterLayerTable> tables,
      DeepSeekExpertPlanStore& store,
      DeepSeekLearnedRouterOperations& operations);

  Status append_plan_work(
      std::span<const std::uint32_t> token_ids,
      std::vector<DeepSeekHashRouterLayerScores> layer_scores,
      DeepSeekRankComputeWorkBuilder& builder);
  Status append_projected_plan_work(
      std::span<const std::uint32_t> token_ids,
      std::vector<DeepSeekHashRouterLayerPlanWork> layers,
      DeepSeekRankComputeWorkBuilder& builder);

  [[nodiscard]] std::uint32_t owned_hash_layer_count() const noexcept {
    return owned_hash_layer_count_;
  }
  [[nodiscard]] DeepSeekStageRange owned_layers() const noexcept {
    return owned_layers_;
  }

 private:
  DeepSeekStageRange owned_layers_;
  std::uint32_t maximum_tokens_ = 0;
  std::uint32_t vocabulary_size_ = 0;
  std::uint32_t owned_hash_layer_count_ = 0;
  std::array<std::shared_ptr<const std::vector<std::uint16_t>>, 3> tables_;
  std::array<std::optional<DeepSeekRouteScratchArena>, 3> scratch_;
  std::array<std::optional<DeepSeekHashRouterCoordinator>, 3> coordinators_;
};

}  // namespace pih
