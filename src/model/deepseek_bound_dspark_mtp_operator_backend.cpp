#include "pih/model/deepseek_bound_dspark_mtp_operator_backend.h"

#include <array>

namespace pih {
namespace {

bool complete(const DeepSeekAttentionWeightBindings& value) {
  return value.wq_a_fp8 != 0 && value.wq_a_scale_ue8m0 != 0 &&
         value.q_norm_bf16 != 0 && value.wq_b_fp8 != 0 &&
         value.wq_b_scale_ue8m0 != 0 && value.wkv_fp8 != 0 &&
         value.wkv_scale_ue8m0 != 0 && value.kv_norm_bf16 != 0 &&
         value.attention_sink_f32 != 0 && value.wo_a_fp8 != 0 &&
         value.wo_a_scale_ue8m0 != 0 && value.wo_b_fp8 != 0 &&
         value.wo_b_scale_ue8m0 != 0 && value.generation != 0;
}

bool complete(const DeepSeekMhcWeightBindings& value) {
  return value.attention_norm_bf16 != 0 && value.attention_fn_f32 != 0 &&
         value.attention_scale_f32 != 0 && value.attention_base_f32 != 0 &&
         value.feed_forward_norm_bf16 != 0 &&
         value.feed_forward_fn_f32 != 0 &&
         value.feed_forward_scale_f32 != 0 &&
         value.feed_forward_base_f32 != 0 && value.generation != 0;
}

bool complete(const DeepSeekExpertMatrixDeviceView& value) {
  return value.packed.address != 0 &&
         value.packed.bytes ==
             DeepSeekExpertBundleLayout::kPackedBytesPerMatrix &&
         value.scales.address != 0 &&
         value.scales.bytes ==
             DeepSeekExpertBundleLayout::kScaleBytesPerMatrix;
}

bool complete(const DeepSeekAttentionProjectionDeviceView& value) {
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

bool complete(const DeepSeekMhcDeviceView& value) {
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
         // One resident expert is launched at a time, so a subwave can carry
         // at most one route per draft token even though each token selects
         // six different experts across the complete route plan.
         fits(value.route_weights_f32, tokens * sizeof(float)) &&
         fits(value.token_indices_u32, tokens * sizeof(std::uint32_t)) &&
         fits(value.error_flag_u32, sizeof(std::uint32_t));
}

bool valid(const DeepSeekDsparkMtpStageResources& value) {
  if (!is_valid_deepseek_dspark_stage(value.stage) || value.weights == nullptr ||
      value.resident_experts == nullptr || value.state_layout == nullptr ||
      value.transaction == nullptr || value.prepare_epoch == 0 ||
      value.weight_generation == 0 ||
      value.weights->stage != value.stage ||
      value.weights->generation != value.weight_generation ||
      value.resident_experts->generation() != value.weight_generation ||
      value.recent_state.stage != value.stage ||
      value.recent_state.recent_bf16.address == 0 ||
      value.recent_state.recent_bf16.bytes !=
          kDeepSeekDsparkRecentStateBytes ||
      !complete(value.weights->attention) ||
      value.weights->attention.generation != value.weight_generation ||
      !complete(value.weights->mhc) ||
      value.weights->mhc.generation != value.weight_generation ||
      value.weights->router_weight_bf16 == 0 ||
      value.weights->router_bias_f32 == 0 ||
      !complete(value.weights->shared_expert.w1) ||
      !complete(value.weights->shared_expert.w2) ||
      !complete(value.weights->shared_expert.w3) ||
      !complete(value.attention_workspace) || !complete(value.mhc_workspace) ||
      value.stage_input_hc_bf16 == 0 ||
      value.attention_output_hc_bf16 == 0 ||
      value.stage_output_hc_bf16 == 0 ||
      value.stage_input_hc_bf16 == value.attention_output_hc_bf16 ||
      value.attention_output_hc_bf16 == value.stage_output_hc_bf16 ||
      value.main_normalized_bf16 == 0 ||
      value.main_activation_e4m3 == 0 ||
      value.main_activation_scale_ue8m0 == 0 || value.main_kv_bf16 == 0 ||
      value.main_positions_u32 == 0 || value.draft_positions_u32 == 0 ||
      value.rope_frequencies_f32 == 0 || value.error_flag_u32 == 0 ||
      value.stream == 0 || value.completion_event == 0 ||
      value.router_scores_f32 == 0 ||
      value.router_host_scores.size() != 5U * 256U ||
      value.router_host_bias.size() != 256U ||
      !complete(value.expert_arena) || value.expert_accumulator_f32 == 0 ||
      value.expert_kernel == nullptr || value.position_table_count == 0 ||
      value.current_position > UINT32_MAX - 5U ||
      value.current_position + 5U >= value.position_table_count) {
    return false;
  }
  if (value.transaction->state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      value.transaction->prepare_epoch() != value.prepare_epoch ||
      value.transaction->stream() == 0 ||
      value.transaction->stream() != value.stream) {
    return false;
  }
  auto bank = value.transaction->tentative_fixed_state();
  if (!bank.ok()) return false;
  auto expected = value.state_layout->ResolveDspark(value.stage, *bank);
  return expected.ok() && expected->stage == value.recent_state.stage &&
         expected->recent_bf16.address ==
             value.recent_state.recent_bf16.address &&
         expected->recent_bf16.bytes == value.recent_state.recent_bf16.bytes;
}

bool valid(const DeepSeekDsparkMtpPrefillResources& value,
           const DeepSeekPipelinePlanDescriptor& plan) {
  if (plan.phase != DeepSeekPlanPhase::kPrefill || plan.sequence_count != 1 ||
      plan.token_count != value.token_count ||
      !is_valid_deepseek_dspark_stage(value.stage) ||
      value.weights.wkv_fp8 == 0 || value.weights.wkv_scale_ue8m0 == 0 ||
      value.weights.kv_norm_bf16 == 0 || value.weights.generation == 0 ||
      value.weights.generation != value.weight_generation ||
      value.state_layout == nullptr || value.transaction == nullptr ||
      value.prepare_epoch == 0 || value.weight_generation == 0 ||
      value.recent_state.stage != value.stage ||
      value.recent_state.recent_bf16.address == 0 ||
      value.recent_state.recent_bf16.bytes !=
          kDeepSeekDsparkRecentStateBytes ||
      value.main_normalized_bf16 == 0 || value.activation_e4m3 == 0 ||
      value.activation_scale_ue8m0 == 0 || value.positions_u32 == 0 ||
      value.kv_scratch_bf16 == 0 || value.rope_frequencies_f32 == 0 ||
      value.error_flag_u32 == 0 || value.stream == 0 ||
      value.completion_event == 0 ||
      value.token_count == 0 ||
      value.token_count > value.position_table_count) {
    return false;
  }
  if (value.transaction->state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      value.transaction->prepare_epoch() != value.prepare_epoch ||
      value.transaction->stream() != value.stream) {
    return false;
  }
  auto bank = value.transaction->tentative_fixed_state();
  if (!bank.ok()) return false;
  auto expected = value.state_layout->ResolveDspark(value.stage, *bank);
  return expected.ok() && expected->stage == value.recent_state.stage &&
         expected->recent_bf16.address ==
             value.recent_state.recent_bf16.address &&
         expected->recent_bf16.bytes == value.recent_state.recent_bf16.bytes;
}

bool valid(const DeepSeekPipelinePlanDescriptor& plan) {
  return plan.engine_epoch != 0 && plan.plan_sequence != 0 &&
         plan.token_count != 0 && plan.sequence_count != 0 &&
         ((plan.phase == DeepSeekPlanPhase::kPrefill &&
           plan.sequence_count == 1) ||
          (plan.phase == DeepSeekPlanPhase::kDecode &&
           plan.token_count == 1 && plan.sequence_count == 1));
}

}  // namespace

Result<DeepSeekBoundDsparkMtpOperatorBackend>
DeepSeekBoundDsparkMtpOperatorBackend::Create(
    DeepSeekDsparkMtpStageWorkProvider& provider) {
  DeepSeekBoundDsparkMtpOperatorBackend result;
  result.provider_ = &provider;
  return result;
}

Status DeepSeekBoundDsparkMtpOperatorBackend::poison(
    Status status, bool cancel_operation) {
  Status cancel = Status::Ok();
  if (cancel_operation && operation_ != nullptr) {
    cancel = operation_->cancel();
  }
  work_ = nullptr;
  operation_ = nullptr;
  state_ = State::kPoisoned;
  if (!status.ok()) return status;
  if (!cancel.ok()) return cancel;
  return Status::Internal("DeepSeek bound DSpark MTP backend poisoned");
}

Status DeepSeekBoundDsparkMtpOperatorBackend::launch(
    const DeepSeekDsparkMtpOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (state_ != State::kIdle ||
      !is_valid_deepseek_dspark_stage(command.stage) ||
      (command.kind != DeepSeekDsparkMtpOperatorKind::kAttention &&
       command.kind != DeepSeekDsparkMtpOperatorKind::kMoe) || !valid(plan)) {
    return Status::FailedPrecondition(
        "DeepSeek bound DSpark MTP backend is not launchable");
  }
  if (plan.phase == DeepSeekPlanPhase::kPrefill &&
      command.kind != DeepSeekDsparkMtpOperatorKind::kAttention) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark prefill cannot launch a MoE operation");
  }
  auto work = provider_->resolve(command.stage, plan);
  if (!work.ok() || *work == nullptr) {
    return poison(work.ok()
                      ? Status::FailedPrecondition(
                            "DeepSeek DSpark MTP stage work is unavailable")
                      : work.status(),
                  false);
  }
  work_ = *work;
  const bool prefill = plan.phase == DeepSeekPlanPhase::kPrefill;
  const bool resource_identity = prefill
      ? valid(work_->prefill_resources, plan) &&
            work_->prefill_resources.stage == command.stage
      : valid(work_->resources) && work_->resources.stage == command.stage;
  const bool operation_identity = work_->attention != nullptr &&
      (prefill ? (work_->moe == nullptr || work_->attention != work_->moe)
               : (work_->moe != nullptr && work_->attention != work_->moe));
  if (!resource_identity || !operation_identity) {
    return poison(Status::InvalidArgument(
                      "DeepSeek DSpark MTP stage work identity is invalid"),
                  false);
  }
  operation_ = command.kind == DeepSeekDsparkMtpOperatorKind::kAttention
                   ? work_->attention
                   : work_->moe;
  const auto status = prefill
      ? operation_->launch_prefill(plan, work_->prefill_resources)
      : operation_->launch(plan, work_->resources);
  if (!status.ok()) return poison(status, true);
  state_ = State::kInflight;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus>
DeepSeekBoundDsparkMtpOperatorBackend::poll() {
  if (state_ != State::kInflight || operation_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek bound DSpark MTP backend is not pollable");
  }
  auto result = operation_->poll();
  if (!result.ok()) return poison(result.status(), true);
  if (*result == DeepSeekStageComputeStatus::kError) {
    const auto status = poison(
        Status::Internal("DeepSeek DSpark MTP stage operation failed"), true);
    if (!status.ok() && status.code() != StatusCode::kInternal) return status;
    return DeepSeekStageComputeStatus::kError;
  }
  if (*result == DeepSeekStageComputeStatus::kSuccess) {
    work_ = nullptr;
    operation_ = nullptr;
    state_ = State::kIdle;
  }
  return *result;
}

Status DeepSeekBoundDsparkMtpOperatorBackend::cancel() {
  if (state_ != State::kInflight || operation_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek bound DSpark MTP backend has no cancellable operation");
  }
  const auto status = operation_->cancel();
  work_ = nullptr;
  operation_ = nullptr;
  state_ = State::kPoisoned;
  return status;
}

}  // namespace pih
