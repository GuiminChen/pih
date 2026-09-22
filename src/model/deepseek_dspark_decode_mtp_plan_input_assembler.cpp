#include "pih/model/deepseek_dspark_decode_mtp_plan_input_assembler.h"

#include <algorithm>

namespace pih {
namespace {

bool nonzero(const DeepSeekAttentionProjectionDeviceView& value) {
  const std::array addresses{
      value.input_e4m3, value.input_scale_ue8m0, value.q_a_bf16,
      value.q_norm_bf16, value.q_e4m3, value.q_scale_ue8m0,
      value.query_bf16, value.kv_bf16, value.attention_output_bf16,
      value.wo_a_activation_e4m3, value.wo_a_activation_scale_ue8m0,
      value.wo_a_output_bf16, value.wo_b_activation_e4m3,
      value.wo_b_activation_scale_ue8m0, value.branch_output_bf16,
      value.token_ids_u32, value.positions_u32, value.error_flag_u32};
  return std::ranges::all_of(
      addresses, [](std::uintptr_t address) { return address != 0; });
}

bool nonzero(const DeepSeekMhcDeviceView& value) {
  const std::array addresses{
      value.residual_a_bf16, value.residual_b_bf16,
      value.layer_input_bf16, value.ffn_branch_output_bf16,
      value.post_mix_f32, value.residual_mix_f32};
  return std::ranges::all_of(
      addresses, [](std::uintptr_t address) { return address != 0; });
}

bool fits(const DeepSeekExpertArenaSpan& span, std::uint64_t bytes) {
  return span.address != 0 && span.bytes >= bytes;
}

bool complete(const DeepSeekExpertComputeArena& value) {
  constexpr std::uint64_t tokens = 5;
  return fits(value.route_input_bf16, tokens * 4096U * 2U) &&
         value.expert_output_bf16.address == value.route_input_bf16.address &&
         value.expert_output_bf16.bytes == value.route_input_bf16.bytes &&
         fits(value.activation_e4m3, tokens * 4096U) &&
         fits(value.activation_scale_bits, tokens * 32U) &&
         fits(value.gate_or_middle_bf16, tokens * 2048U * 2U) &&
         fits(value.up_bf16, tokens * 2048U * 2U) &&
         fits(value.route_weights_f32, tokens * sizeof(float)) &&
         fits(value.token_indices_u32, tokens * sizeof(std::uint32_t)) &&
         fits(value.error_flag_u32, sizeof(std::uint32_t));
}

bool distinct_stage_buffers(const DeepSeekDsparkDecodeMtpPlanInput& input) {
  const std::array buffers{
      input.device.draft_input_hc_bf16,
      input.device.stage_residual_a_bf16,
      input.device.stage_residual_b_bf16,
      input.mhc_workspace.residual_b_bf16};
  for (std::size_t left = 0; left < buffers.size(); ++left) {
    for (std::size_t right = left + 1; right < buffers.size(); ++right) {
      if (buffers[left] == buffers[right]) return false;
    }
  }
  return true;
}

}  // namespace

Result<std::vector<DeepSeekBoundDsparkMtpStageWork>>
DeepSeekDsparkDecodeMtpPlanInputAssembler::Assemble(
    const DeepSeekDsparkDecodeMtpPlanInput& input) {
  if (input.weights == nullptr || input.resident_experts == nullptr ||
      input.state_layout == nullptr || input.weights->generation() == 0 ||
      input.resident_experts->generation() != input.weights->generation() ||
      input.device.draft_input_hc_bf16 == 0 ||
      input.device.stage_residual_a_bf16 == 0 ||
      input.device.stage_residual_b_bf16 == 0 ||
      input.device.main_normalized_bf16 == 0 ||
      input.device.main_quant_fp8 == 0 ||
      input.device.main_quant_scale_ue8m0 == 0 ||
      input.device.prefill_kv_bf16 == 0 ||
      input.device.draft_positions_u32 == 0 ||
      input.device.error_flag_u32 == 0 ||
      !nonzero(input.attention_workspace) || !nonzero(input.mhc_workspace) ||
      !distinct_stage_buffers(input) ||
      input.rope_frequencies_f32 == 0 || input.router_scores_f32 == 0 ||
      input.router_host_scores.size() != 5U * 256U ||
      input.router_host_scores.data() == nullptr ||
      input.router_host_bias.size() != 256U ||
      input.router_host_bias.data() == nullptr ||
      !complete(input.expert_arena) ||
      input.expert_accumulator_f32 == 0 || input.expert_kernel == nullptr ||
      input.stream == 0 || input.completion_event == 0 ||
      input.position_table_count == 0 ||
      input.current_position > UINT32_MAX - 5U ||
      input.current_position + 5U >= input.position_table_count) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode MTP plan input is incomplete");
  }

  const std::array<std::uintptr_t, kDeepSeekDsparkStageCount> stage_inputs{
      input.device.draft_input_hc_bf16,
      input.device.stage_residual_a_bf16,
      input.device.stage_residual_b_bf16};
  const std::array<std::uintptr_t, kDeepSeekDsparkStageCount> stage_outputs{
      input.device.stage_residual_a_bf16,
      input.device.stage_residual_b_bf16,
      input.device.draft_input_hc_bf16};
  std::vector<DeepSeekBoundDsparkMtpStageWork> result;
  result.reserve(kDeepSeekDsparkStageCount);
  for (std::size_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    const auto stage = static_cast<DeepSeekDsparkStageId>(index);
    const auto* attention = input.attention_operations[index];
    const auto* moe = input.moe_operations[index];
    const auto& common = input.weights->common(stage);
    if (attention == nullptr || moe == nullptr || attention == moe ||
        common.stage != stage ||
        common.generation != input.weights->generation()) {
      return Status::FailedPrecondition(
          "DeepSeek DSpark decode stage operation is incomplete");
    }

    DeepSeekBoundDsparkMtpStageWork work;
    work.attention = input.attention_operations[index];
    work.moe = input.moe_operations[index];
    auto& resources = work.resources;
    resources.stage = stage;
    resources.weights = &common;
    resources.resident_experts = input.resident_experts;
    resources.state_layout = input.state_layout;
    resources.weight_generation = input.weights->generation();
    resources.attention_workspace = input.attention_workspace;
    resources.mhc_workspace = input.mhc_workspace;
    resources.stage_input_hc_bf16 = stage_inputs[index];
    resources.attention_output_hc_bf16 =
        input.mhc_workspace.residual_b_bf16;
    resources.stage_output_hc_bf16 = stage_outputs[index];
    resources.main_normalized_bf16 = input.device.main_normalized_bf16;
    resources.main_activation_e4m3 = input.device.main_quant_fp8;
    resources.main_activation_scale_ue8m0 =
        input.device.main_quant_scale_ue8m0;
    resources.main_kv_bf16 = input.device.prefill_kv_bf16;
    resources.main_positions_u32 = input.attention_workspace.positions_u32;
    resources.draft_positions_u32 = input.device.draft_positions_u32;
    resources.rope_frequencies_f32 = input.rope_frequencies_f32;
    resources.error_flag_u32 = input.device.error_flag_u32;
    resources.stream = input.stream;
    resources.completion_event = input.completion_event;
    resources.router_scores_f32 = input.router_scores_f32;
    resources.router_host_scores = input.router_host_scores;
    resources.router_host_bias = input.router_host_bias;
    resources.expert_arena = input.expert_arena;
    resources.expert_accumulator_f32 = input.expert_accumulator_f32;
    resources.expert_kernel = input.expert_kernel;
    resources.current_position = input.current_position;
    resources.position_table_count = input.position_table_count;
    result.push_back(work);
  }
  return result;
}

}  // namespace pih
