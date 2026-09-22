#pragma once

#include <cstdint>
#include <vector>

#include "pih/model/deepseek_expert_subwave_plan.h"
#include "pih/model/deepseek_expert_plan_catalog.h"
#include "pih/model/deepseek_route_scratch_arena.h"

namespace pih {

class DeepSeekHashRouter final {
 public:
  static constexpr float kRoutedScalingFactor = 1.5F;

  static Result<DeepSeekExpertSubwavePlan> Route(
      const std::vector<std::uint32_t>& token_ids,
      const std::vector<float>& raw_scores, std::uint32_t vocabulary_size,
      const std::vector<std::uint16_t>& token_to_experts);
  static Status RouteInto(
      std::uint32_t layer, std::span<const std::uint32_t> token_ids,
      std::span<const float> raw_scores, std::uint32_t vocabulary_size,
      std::span<const std::uint16_t> token_to_experts,
      DeepSeekRouteScratchArena& scratch,
      DeepSeekExpertPlanStore& store);
};

}  // namespace pih
