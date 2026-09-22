#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "pih/model/deepseek_expert_subwave_plan.h"
#include "pih/model/deepseek_expert_plan_catalog.h"
#include "pih/model/deepseek_route_scratch_arena.h"

namespace pih {

class DeepSeekLearnedRouter final {
 public:
  static constexpr float kRoutedScalingFactor = 1.5F;

  // Flash-0731 uses ungrouped sqrt(softplus) routing over all 256 experts.
  // Some generic DeepSeek loaders expose legacy n_group/topk_group fields;
  // those values must not turn this compatibility path into grouped top-k.
  // Selection bias affects the Top-6 membership only, while normalization
  // uses the un-biased base scores, exactly as encoded below.
  static Result<DeepSeekExpertSubwavePlan> Route(
      std::uint32_t token_count, const std::vector<float>& raw_scores,
      const std::array<float, DeepSeekExpertSubwavePlan::kExpertCount>& bias);
  static Status RouteInto(
      std::uint32_t layer, std::uint32_t token_count,
      std::span<const float> raw_scores,
      const std::array<float, DeepSeekExpertSubwavePlan::kExpertCount>& bias,
      DeepSeekRouteScratchArena& scratch,
      DeepSeekExpertPlanStore& store);
};

}  // namespace pih
