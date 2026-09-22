#pragma once

#include "pih/model/deepseek_attention_output_projection_coordinator.h"
#include "pih/model/deepseek_attention_projection_coordinator.h"
#include "pih/model/deepseek_attention_projection_device_resources.h"
#include "pih/model/deepseek_attention_weight_bindings.h"

namespace pih {

struct DeepSeekAttentionProjectionLayerSubmissions final {
  DeepSeekAttentionProjectionSubmission input;
  DeepSeekAttentionOutputProjectionSubmission output;
  std::uintptr_t sparse_query_bf16 = 0;
  std::uintptr_t sparse_kv_bf16 = 0;
  std::uintptr_t sparse_output_bf16 = 0;
  std::uintptr_t branch_output_bf16 = 0;
};

class DeepSeekAttentionProjectionSubmissionAssembler final {
 public:
  static Result<DeepSeekAttentionProjectionLayerSubmissions> Assemble(
      const DeepSeekAttentionWeightBindings& weights,
      DeepSeekAttentionProjectionDeviceView workspace,
      std::uint32_t maximum_tokens, std::uint32_t token_count,
      std::uintptr_t layer_input_bf16,
      std::uintptr_t sparse_output_bf16,
      std::uintptr_t frequencies_f32,
      std::uint32_t table_position_count, std::uintptr_t stream);
};

}  // namespace pih
