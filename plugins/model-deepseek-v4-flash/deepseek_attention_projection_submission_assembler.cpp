#include "pih/model/deepseek_attention_projection_submission_assembler.h"

#include <algorithm>
#include <array>

namespace pih {
namespace {

bool complete(const DeepSeekAttentionWeightBindings& weights) {
  const std::array values{
      weights.wq_a_fp8, weights.wq_a_scale_ue8m0, weights.q_norm_bf16,
      weights.wq_b_fp8, weights.wq_b_scale_ue8m0, weights.wkv_fp8,
      weights.wkv_scale_ue8m0, weights.kv_norm_bf16,
      weights.attention_sink_f32, weights.wo_a_fp8,
      weights.wo_a_scale_ue8m0, weights.wo_b_fp8,
      weights.wo_b_scale_ue8m0};
  return weights.generation != 0 &&
         std::ranges::all_of(values, [](auto value) { return value != 0; });
}

bool complete(DeepSeekAttentionProjectionDeviceView workspace) {
  const std::array values{
      workspace.input_e4m3, workspace.input_scale_ue8m0,
      workspace.q_a_bf16, workspace.q_norm_bf16, workspace.q_e4m3,
      workspace.q_scale_ue8m0, workspace.query_bf16, workspace.kv_bf16,
      workspace.attention_output_bf16,
      workspace.wo_a_activation_e4m3,
      workspace.wo_a_activation_scale_ue8m0,
      workspace.wo_a_output_bf16, workspace.wo_b_activation_e4m3,
      workspace.wo_b_activation_scale_ue8m0,
      workspace.branch_output_bf16, workspace.token_ids_u32,
      workspace.positions_u32,
      workspace.error_flag_u32};
  return std::ranges::all_of(values,
                             [](auto value) { return value != 0; });
}

Status validate(
    const DeepSeekAttentionProjectionLayerSubmissions& submissions) {
  const auto& input = submissions.input;
  const auto& output = submissions.output;
  for (const auto status : {
           validate_deepseek_fp8_activation_quant_launch(input.input_quant),
           validate_deepseek_fp8_gemm_launch(input.wq_a),
           validate_deepseek_rms_norm_launch(input.q_norm),
           validate_deepseek_fp8_activation_quant_launch(input.q_quant),
           validate_deepseek_fp8_gemm_launch(input.wq_b),
           validate_deepseek_head_rms_launch(input.q_head_rms),
           validate_deepseek_rotary_launch(input.q_rope),
           validate_deepseek_fp8_gemm_launch(input.wkv),
           validate_deepseek_rms_norm_launch(input.kv_norm),
           validate_deepseek_rotary_launch(input.kv_rope),
           validate_deepseek_kv_fp8_simulate_launch(input.kv_simulate),
           validate_deepseek_rotary_launch(output.inverse_rope),
           validate_deepseek_grouped_fp8_gemm_launch(output.wo_a),
           validate_deepseek_fp8_activation_quant_launch(output.quant),
           validate_deepseek_fp8_gemm_launch(output.wo_b)}) {
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekAttentionProjectionLayerSubmissions>
DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
    const DeepSeekAttentionWeightBindings& weights,
    DeepSeekAttentionProjectionDeviceView workspace,
    std::uint32_t maximum_tokens, std::uint32_t token_count,
    std::uintptr_t layer_input_bf16,
    std::uintptr_t sparse_output_bf16,
    std::uintptr_t frequencies_f32,
    std::uint32_t table_position_count, std::uintptr_t stream) {
  if (!complete(weights) || !complete(workspace) || maximum_tokens == 0 ||
      maximum_tokens > 4096 || token_count == 0 ||
      token_count > maximum_tokens || layer_input_bf16 == 0 ||
      sparse_output_bf16 == 0 || frequencies_f32 == 0 ||
      workspace.positions_u32 == 0 || table_position_count == 0 ||
      table_position_count > 1048576 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek attention projection submission input is invalid");
  }

  const auto error = workspace.error_flag_u32;
  DeepSeekAttentionProjectionLayerSubmissions result;
  result.sparse_query_bf16 = workspace.query_bf16;
  result.sparse_kv_bf16 = workspace.kv_bf16;
  result.sparse_output_bf16 = sparse_output_bf16;
  result.branch_output_bf16 = workspace.branch_output_bf16;
  result.input = {
      {layer_input_bf16, workspace.input_e4m3,
       workspace.input_scale_ue8m0, error, stream, token_count, 4096},
      {workspace.input_e4m3, workspace.input_scale_ue8m0,
       weights.wq_a_fp8, weights.wq_a_scale_ue8m0, workspace.q_a_bf16,
       error, stream, token_count, 1024, 4096},
      {workspace.q_a_bf16, weights.q_norm_bf16, workspace.q_norm_bf16,
       error, stream, token_count, 1024, DeepSeekRmsNormLaunch::kEpsilon},
      {workspace.q_norm_bf16, workspace.q_e4m3,
       workspace.q_scale_ue8m0, error, stream, token_count, 1024},
      {workspace.q_e4m3, workspace.q_scale_ue8m0, weights.wq_b_fp8,
       weights.wq_b_scale_ue8m0, workspace.query_bf16, error, stream,
       token_count, 32768, 1024},
      {workspace.query_bf16, workspace.query_bf16, error, stream,
       token_count, 64, 512, DeepSeekRmsNormLaunch::kEpsilon},
      {workspace.query_bf16, frequencies_f32, error, stream, token_count,
       64, 512, 64, false, workspace.positions_u32, table_position_count},
      {workspace.input_e4m3, workspace.input_scale_ue8m0, weights.wkv_fp8,
       weights.wkv_scale_ue8m0, workspace.kv_bf16, error, stream,
       token_count, 512, 4096},
      {workspace.kv_bf16, weights.kv_norm_bf16, workspace.kv_bf16,
       error, stream, token_count, 512, DeepSeekRmsNormLaunch::kEpsilon},
      {workspace.kv_bf16, frequencies_f32, error, stream, token_count,
       1, 512, 64, false, workspace.positions_u32, table_position_count},
      {workspace.kv_bf16, error, stream, token_count, 512, 448, 64}};
  result.output = {
      {sparse_output_bf16, frequencies_f32, error, stream, token_count,
       64, 512, 64, true, workspace.positions_u32, table_position_count},
      {sparse_output_bf16, workspace.wo_a_activation_e4m3,
       workspace.wo_a_activation_scale_ue8m0, weights.wo_a_fp8,
       weights.wo_a_scale_ue8m0, workspace.wo_a_output_bf16, error,
       stream, token_count, 8, 1024, 4096},
      {workspace.wo_a_output_bf16, workspace.wo_b_activation_e4m3,
       workspace.wo_b_activation_scale_ue8m0, error, stream, token_count,
       8192},
      {workspace.wo_b_activation_e4m3,
       workspace.wo_b_activation_scale_ue8m0, weights.wo_b_fp8,
       weights.wo_b_scale_ue8m0, workspace.branch_output_bf16, error,
       stream, token_count, 4096, 8192}};
  auto status = validate(result);
  if (!status.ok()) return status;
  return result;
}

}  // namespace pih
