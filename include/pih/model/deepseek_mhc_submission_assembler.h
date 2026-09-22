#pragma once

#include "pih/model/deepseek_mhc_device_resources.h"
#include "pih/model/deepseek_mhc_sequence_executor.h"
#include "pih/model/deepseek_mhc_weight_bindings.h"

namespace pih {

struct DeepSeekMhcLayerSubmissions final {
  DeepSeekMhcSequenceSubmission attention;
  DeepSeekMhcSequenceSubmission feed_forward;
  std::uintptr_t layer_input_bf16 = 0;
  std::uintptr_t residual_output_bf16 = 0;
};

class DeepSeekMhcSubmissionAssembler final {
 public:
  static Result<DeepSeekMhcLayerSubmissions> Assemble(
      std::uint32_t layer, const DeepSeekMhcWeightBindings& weights,
      DeepSeekMhcDeviceView workspace, std::uint32_t maximum_tokens,
      std::uint32_t token_count, std::uintptr_t residual_input_bf16,
      std::uintptr_t attention_branch_output_bf16,
      std::uintptr_t device_error_flag_u32, std::uintptr_t stream);
};

}  // namespace pih
