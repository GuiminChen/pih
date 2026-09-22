#include "pih/model/deepseek_dense_mhc_layer_submission_assembler.h"

namespace pih {

Result<DeepSeekDenseMhcAssembledLayer>
DeepSeekDenseMhcLayerSubmissionAssembler::Assemble(
    std::uint32_t layer,
    const DeepSeekAttentionWeightBindings& attention_weights,
    const DeepSeekMhcWeightBindings& mhc_weights,
    const DeepSeekAttentionProjectionDeviceResources& attention_resources,
    const DeepSeekMhcDeviceResources& mhc_resources,
    std::uint32_t token_count, std::uintptr_t residual_input_bf16,
    std::uintptr_t frequencies_f32,
    std::uint32_t table_position_count, std::uintptr_t stream) {
  if (attention_weights.generation != mhc_weights.generation ||
      attention_resources.maximum_tokens() != mhc_resources.maximum_tokens() ||
      attention_resources.context_identity() !=
          mhc_resources.context_identity() ||
      attention_resources.device_ordinal() != mhc_resources.device_ordinal()) {
    return Status::FailedPrecondition(
        "DeepSeek dense mHC device resource identities differ");
  }
  const auto attention_view = attention_resources.view();
  const auto mhc_view = mhc_resources.view();
  auto mhc = DeepSeekMhcSubmissionAssembler::Assemble(
      layer, mhc_weights, mhc_view, mhc_resources.maximum_tokens(),
      token_count, residual_input_bf16,
      attention_view.branch_output_bf16, attention_view.error_flag_u32,
      stream);
  if (!mhc.ok()) return mhc.status();
  auto attention = DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
      attention_weights, attention_view,
      attention_resources.maximum_tokens(), token_count,
      mhc->layer_input_bf16, attention_view.attention_output_bf16,
      frequencies_f32, table_position_count, stream);
  if (!attention.ok()) return attention.status();
  if (attention->branch_output_bf16 !=
          mhc->attention.branch_output_bf16 ||
      attention->input.input_quant.error_flag !=
          mhc->attention.device_error_flag_u32 ||
      attention->input.input_quant.stream != mhc->attention.stream ||
      attention->input.input_quant.token_count !=
          mhc->attention.token_count) {
    return Status::Internal(
        "DeepSeek dense mHC component assemblers disagree");
  }
  DeepSeekDenseMhcAssembledLayer result;
  result.input.layer = layer;
  result.input.attention_input = attention->input;
  result.input.attention_output = attention->output;
  result.input.sparse_query_bf16 = attention->sparse_query_bf16;
  result.input.sparse_kv_bf16 = attention->sparse_kv_bf16;
  result.input.sparse_output_bf16 = attention->sparse_output_bf16;
  result.input.mhc_attention = mhc->attention;
  result.input.mhc_feed_forward = mhc->feed_forward;
  result.residual_output_bf16 = mhc->residual_output_bf16;
  return result;
}

}  // namespace pih
