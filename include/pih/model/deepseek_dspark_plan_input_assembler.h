#pragma once

#include "pih/model/deepseek_dspark_device_resources.h"
#include "pih/model/deepseek_dspark_stage_backend.h"
#include "pih/model/deepseek_dspark_weight_bindings.h"
#include "pih/model/deepseek_endpoint_weight_bindings.h"

namespace pih {

class DeepSeekDsparkPlanInputAssembler final {
 public:
  static Result<DeepSeekDsparkStageWork> Assemble(
      DeepSeekStagePlan stage, std::uintptr_t input_token_ids_u32,
      const DeepSeekEndpointWeightBindings& endpoint_weights,
      const DeepSeekDsparkWeightBindings& weights,
      DeepSeekDsparkDeviceView device,
      DeepSeekDsparkEmbedCoordinator* embed_coordinator,
      DeepSeekDsparkHeadExecutor* head_executor,
      DeepSeekAttentionSequenceTransaction* unbound_transaction,
      std::uintptr_t stream, std::uint32_t maximum_tokens);
  static Result<DeepSeekDsparkStageWork> AssemblePrefill(
      DeepSeekStagePlan stage, std::uint32_t token_count,
      const DeepSeekDsparkWeightBindings& weights,
      DeepSeekDsparkDeviceView device,
      DeepSeekDsparkEmbedCoordinator* embed_coordinator,
      DeepSeekAttentionSequenceTransaction* unbound_transaction,
      std::uintptr_t stream, std::uint32_t maximum_tokens);
};

}  // namespace pih
