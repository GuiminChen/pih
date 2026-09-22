#pragma once

#include <vector>
#include <span>

#include "pih/model/deepseek_learned_router_device_resources.h"
#include "pih/model/deepseek_learned_router_staging_pool.h"
#include "pih/model/deepseek_learned_router_weight_bindings.h"
#include "pih/model/deepseek_learned_router_work_factory.h"

namespace pih {

struct DeepSeekLearnedRouterLayerInput final {
  std::uint32_t layer = 0;
  std::uintptr_t input_bf16 = 0;
};

class DeepSeekLearnedRouterPlanInputAssembler final {
 public:
  static Result<std::vector<DeepSeekLearnedRouterLayerPlanWork>> Assemble(
      std::uint32_t token_count,
      std::span<const DeepSeekLearnedRouterLayerInput> layer_inputs,
      const DeepSeekLearnedRouterWeightBindings& weights,
      DeepSeekLearnedRouterDeviceView device,
      DeepSeekLearnedRouterStagingPool& staging_pool,
      std::uintptr_t stream, std::uintptr_t completion_event,
      std::uint32_t maximum_tokens);
};

}  // namespace pih
