#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_hash_router_staging_pool.h"
#include "pih/model/deepseek_hash_router_weight_bindings.h"
#include "pih/model/deepseek_hash_router_work_factory.h"
#include "pih/model/deepseek_learned_router_device_resources.h"
#include "pih/model/deepseek_learned_router_plan_input_assembler.h"

namespace pih {

class DeepSeekHashRouterPlanInputAssembler final {
 public:
  static Result<std::vector<DeepSeekHashRouterLayerPlanWork>> Assemble(
      std::uint32_t token_count,
      std::span<const DeepSeekLearnedRouterLayerInput> layer_inputs,
      const DeepSeekHashRouterWeightBindings& weights,
      DeepSeekLearnedRouterDeviceView device,
      DeepSeekHashRouterStagingPool& staging_pool,
      std::uintptr_t stream, std::uintptr_t completion_event,
      std::uint32_t maximum_tokens);
};

}  // namespace pih
