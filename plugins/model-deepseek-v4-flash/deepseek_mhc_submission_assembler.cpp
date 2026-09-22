#include "pih/model/deepseek_mhc_submission_assembler.h"

#include <algorithm>
#include <array>

namespace pih {
namespace {

constexpr float kEpsilon = 1.0e-6F;
constexpr std::uint32_t kSinkhornIterations = 20;

bool complete(const DeepSeekMhcWeightBindings& weights) {
  const std::array values{
      weights.attention_norm_bf16, weights.attention_fn_f32,
      weights.attention_scale_f32, weights.attention_base_f32,
      weights.feed_forward_norm_bf16, weights.feed_forward_fn_f32,
      weights.feed_forward_scale_f32, weights.feed_forward_base_f32};
  return weights.generation != 0 &&
         std::ranges::all_of(values, [](auto value) { return value != 0; });
}

bool complete(DeepSeekMhcDeviceView workspace) {
  const std::array values{
      workspace.residual_a_bf16, workspace.residual_b_bf16,
      workspace.layer_input_bf16, workspace.ffn_branch_output_bf16,
      workspace.post_mix_f32, workspace.residual_mix_f32};
  return std::ranges::all_of(values,
                             [](auto value) { return value != 0; });
}

}  // namespace

Status validate_deepseek_mhc_sequence_submission(
    const DeepSeekMhcSequenceSubmission& value) {
  if ((value.kind != DeepSeekMhcBranchKind::kAttention &&
       value.kind != DeepSeekMhcBranchKind::kFeedForward) ||
      value.layer_id > 42 || value.layer_input_bf16 == 0 ||
      value.branch_output_bf16 == 0 ||
      value.layer_input_bf16 == value.branch_output_bf16) {
    return Status::InvalidArgument(
        "DeepSeek mHC branch submission is invalid");
  }
  const DeepSeekMhcPreLaunch pre{
      .residual_bf16 = value.residual_bf16,
      .fn_f32 = value.fn_f32,
      .scale_f32 = value.scale_f32,
      .base_f32 = value.base_f32,
      .norm_weight_bf16 = value.norm_weight_bf16,
      .post_mix_f32 = value.post_mix_f32,
      .residual_mix_f32 = value.residual_mix_f32,
      .layer_input_bf16 = value.layer_input_bf16,
      .error_flag_u32 = value.device_error_flag_u32,
      .stream = value.stream,
      .token_count = value.token_count,
      .hidden_size = 4096,
      .rms_epsilon = value.rms_epsilon,
      .pre_epsilon = value.pre_epsilon,
      .sinkhorn_epsilon = value.sinkhorn_epsilon,
      .post_multiplier = 2.0F,
      .sinkhorn_iterations = value.sinkhorn_iterations};
  auto status = validate_deepseek_mhc_pre_launch(pre);
  if (!status.ok()) return status;
  const DeepSeekMhcPostLaunch post{
      value.branch_output_bf16, value.residual_bf16,
      value.post_mix_f32, value.residual_mix_f32, value.output_bf16,
      value.device_error_flag_u32, value.stream, value.token_count, 4096};
  status = validate_deepseek_mhc_post_launch(post);
  if (!status.ok()) return status;
  if (value.target_hidden_bf16 == 0) {
    return value.target_stage_index == 0
        ? Status::Ok()
        : Status::InvalidArgument(
              "DeepSeek mHC target hidden tap index has no destination");
  }
  if (value.kind != DeepSeekMhcBranchKind::kFeedForward ||
      value.layer_id < 40 || value.layer_id > 42 ||
      value.target_stage_index != value.layer_id - 40) {
    return Status::InvalidArgument(
        "DeepSeek mHC target hidden tap identity is invalid");
  }
  return validate_deepseek_mhc_target_hidden_tap_launch(
      {value.output_bf16, value.target_hidden_bf16,
       value.device_error_flag_u32, value.stream, value.token_count,
       4096, 4, 3, value.target_stage_index});
}

Result<DeepSeekMhcLayerSubmissions> DeepSeekMhcSubmissionAssembler::Assemble(
    std::uint32_t layer, const DeepSeekMhcWeightBindings& weights,
    DeepSeekMhcDeviceView workspace, std::uint32_t maximum_tokens,
    std::uint32_t token_count, std::uintptr_t residual_input_bf16,
    std::uintptr_t attention_branch_output_bf16,
    std::uintptr_t device_error_flag_u32, std::uintptr_t stream) {
  if (layer > 42 || !complete(weights) || !complete(workspace) ||
      maximum_tokens == 0 || maximum_tokens > 4096 || token_count == 0 ||
      token_count > maximum_tokens || residual_input_bf16 == 0 ||
      attention_branch_output_bf16 == 0 || device_error_flag_u32 == 0 ||
      stream == 0 || residual_input_bf16 == workspace.residual_b_bf16 ||
      attention_branch_output_bf16 == workspace.residual_b_bf16 ||
      workspace.ffn_branch_output_bf16 == workspace.residual_a_bf16) {
    return Status::InvalidArgument(
        "DeepSeek mHC submission assembly input is invalid");
  }

  DeepSeekMhcLayerSubmissions result;
  result.layer_input_bf16 = workspace.layer_input_bf16;
  result.residual_output_bf16 = workspace.residual_a_bf16;
  result.attention = {
      .kind = DeepSeekMhcBranchKind::kAttention,
      .layer_id = layer,
      .residual_bf16 = residual_input_bf16,
      .fn_f32 = weights.attention_fn_f32,
      .scale_f32 = weights.attention_scale_f32,
      .base_f32 = weights.attention_base_f32,
      .norm_weight_bf16 = weights.attention_norm_bf16,
      .post_mix_f32 = workspace.post_mix_f32,
      .residual_mix_f32 = workspace.residual_mix_f32,
      .layer_input_bf16 = workspace.layer_input_bf16,
      .branch_output_bf16 = attention_branch_output_bf16,
      .output_bf16 = workspace.residual_b_bf16,
      .device_error_flag_u32 = device_error_flag_u32,
      .stream = stream,
      .token_count = token_count,
      .rms_epsilon = kEpsilon,
      .pre_epsilon = kEpsilon,
      .sinkhorn_epsilon = kEpsilon,
      .sinkhorn_iterations = kSinkhornIterations};
  result.feed_forward = {
      .kind = DeepSeekMhcBranchKind::kFeedForward,
      .layer_id = layer,
      .residual_bf16 = workspace.residual_b_bf16,
      .fn_f32 = weights.feed_forward_fn_f32,
      .scale_f32 = weights.feed_forward_scale_f32,
      .base_f32 = weights.feed_forward_base_f32,
      .norm_weight_bf16 = weights.feed_forward_norm_bf16,
      .post_mix_f32 = workspace.post_mix_f32,
      .residual_mix_f32 = workspace.residual_mix_f32,
      .layer_input_bf16 = workspace.layer_input_bf16,
      .branch_output_bf16 = workspace.ffn_branch_output_bf16,
      .output_bf16 = workspace.residual_a_bf16,
      .device_error_flag_u32 = device_error_flag_u32,
      .stream = stream,
      .token_count = token_count,
      .rms_epsilon = kEpsilon,
      .pre_epsilon = kEpsilon,
      .sinkhorn_epsilon = kEpsilon,
      .sinkhorn_iterations = kSinkhornIterations};
  auto status = validate_deepseek_mhc_sequence_submission(result.attention);
  if (!status.ok()) return status;
  status = validate_deepseek_mhc_sequence_submission(result.feed_forward);
  if (!status.ok()) return status;
  return result;
}

}  // namespace pih
