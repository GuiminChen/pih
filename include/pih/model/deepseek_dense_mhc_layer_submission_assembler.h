#pragma once

#include "pih/model/deepseek_attention_projection_submission_assembler.h"
#include "pih/model/deepseek_dense_mhc_plan_input_assembler.h"
#include "pih/model/deepseek_mhc_submission_assembler.h"

namespace pih {

struct DeepSeekDenseMhcAssembledLayer final {
  DeepSeekDenseMhcLayerSubmissionInput input;
  std::uintptr_t residual_output_bf16 = 0;
};

class DeepSeekDenseMhcLayerSubmissionAssembler final {
 public:
  static Result<DeepSeekDenseMhcAssembledLayer> Assemble(
      std::uint32_t layer,
      const DeepSeekAttentionWeightBindings& attention_weights,
      const DeepSeekMhcWeightBindings& mhc_weights,
      const DeepSeekAttentionProjectionDeviceResources& attention_resources,
      const DeepSeekMhcDeviceResources& mhc_resources,
      std::uint32_t token_count, std::uintptr_t residual_input_bf16,
      std::uintptr_t frequencies_f32,
      std::uint32_t table_position_count, std::uintptr_t stream);
};

}  // namespace pih
