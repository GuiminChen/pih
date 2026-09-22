#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_dense_mhc_runtime_resources.h"
#include "pih/model/deepseek_dense_mhc_work_factory.h"

namespace pih {

struct DeepSeekDenseMhcLayerSubmissionInput final {
  std::uint32_t layer = 0;
  DeepSeekAttentionProjectionSubmission attention_input;
  DeepSeekAttentionOutputProjectionSubmission attention_output;
  std::uintptr_t sparse_query_bf16 = 0;
  std::uintptr_t sparse_kv_bf16 = 0;
  std::uintptr_t sparse_output_bf16 = 0;
  DeepSeekMhcSequenceSubmission mhc_attention;
  DeepSeekMhcSequenceSubmission mhc_feed_forward;
};

struct DeepSeekDenseMhcPlanInput final {
  std::vector<DeepSeekDenseAttentionLayerPlanInput> dense_attention;
  std::vector<DeepSeekMhcLayerPlanInput> mhc_attention;
  std::vector<DeepSeekMhcLayerPlanInput> mhc_feed_forward;
};

class DeepSeekDenseMhcPlanInputAssembler final {
 public:
  static Result<DeepSeekDenseMhcPlanInput> Assemble(
      DeepSeekStageRange owned_layers,
      std::span<const DeepSeekDenseMhcLayerSubmissionInput> layers,
      DeepSeekDenseMhcRuntimeResources& runtime,
      DeepSeekAttentionSequenceTransaction& transaction);
};

}  // namespace pih
