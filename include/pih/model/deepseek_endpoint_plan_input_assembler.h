#pragma once

#include "pih/model/deepseek_endpoint_device_resources.h"
#include "pih/model/deepseek_endpoint_stage_backend.h"
#include "pih/model/deepseek_endpoint_weight_bindings.h"
#include "pih/model/deepseek_sampler.h"

namespace pih {

class DeepSeekEndpointPlanInputAssembler final {
 public:
  static Result<DeepSeekEndpointStageSequenceWork> Assemble(
      DeepSeekStagePlan stage, std::uint32_t token_count,
      std::uintptr_t token_ids_u32, std::uintptr_t final_hc_bf16,
      const DeepSeekEndpointWeightBindings& weights,
      DeepSeekEndpointDeviceView device,
      DeepSeekEndpointSequenceExecutor* executor,
      std::uintptr_t stream, std::uint32_t maximum_tokens,
      std::optional<DeepSeekPreparedSamplingInput> sampling = std::nullopt);
};

}  // namespace pih
