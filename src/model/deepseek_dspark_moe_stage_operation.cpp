#include "pih/model/deepseek_dspark_moe_stage_operation.h"

#include <algorithm>
#include <array>
#include <vector>

namespace pih {
namespace {

constexpr float kEpsilon = 1.0e-6F;
constexpr std::uint64_t kAccumulatorBytes =
    DeepSeekDsparkMoeStageOperation::kTokenCount * UINT64_C(4096) *
    sizeof(float);

bool fits(const DeepSeekExpertArenaSpan& span, std::uint64_t bytes) {
  return span.address != 0 && span.bytes >= bytes;
}

bool valid_matrix(const DeepSeekExpertMatrixDeviceView& matrix) {
  return matrix.packed.address != 0 &&
         matrix.packed.bytes ==
             DeepSeekExpertBundleLayout::kPackedBytesPerMatrix &&
         matrix.scales.address != 0 &&
         matrix.scales.bytes ==
             DeepSeekExpertBundleLayout::kScaleBytesPerMatrix;
}

bool valid_resources(const DeepSeekDsparkMtpStageResources& value) {
  constexpr auto tokens = static_cast<std::uint64_t>(
      DeepSeekDsparkMoeStageOperation::kTokenCount);
  return value.weights != nullptr && value.resident_experts != nullptr &&
         value.weights->stage == value.stage &&
         value.weights->generation == value.weight_generation &&
         value.resident_experts->generation() == value.weight_generation &&
         value.weight_generation != 0 &&
         value.weights->router_weight_bf16 != 0 &&
         value.weights->router_bias_f32 != 0 &&
         valid_matrix(value.weights->shared_expert.w1) &&
         valid_matrix(value.weights->shared_expert.w2) &&
         valid_matrix(value.weights->shared_expert.w3) &&
         value.attention_output_hc_bf16 != 0 &&
         value.stage_output_hc_bf16 != 0 &&
         value.attention_output_hc_bf16 != value.stage_output_hc_bf16 &&
         value.mhc_workspace.layer_input_bf16 != 0 &&
         value.mhc_workspace.ffn_branch_output_bf16 != 0 &&
         value.mhc_workspace.post_mix_f32 != 0 &&
         value.mhc_workspace.residual_mix_f32 != 0 &&
         value.router_scores_f32 != 0 &&
         value.router_host_scores.size() == tokens * 256U &&
         value.router_host_bias.size() == 256U &&
         value.expert_accumulator_f32 != 0 && value.expert_kernel != nullptr &&
         value.error_flag_u32 != 0 && value.stream != 0 &&
         value.completion_event != 0 &&
         fits(value.expert_arena.route_input_bf16, tokens * 4096U * 2U) &&
         value.expert_arena.expert_output_bf16.address ==
             value.expert_arena.route_input_bf16.address &&
         fits(value.expert_arena.activation_e4m3, tokens * 4096U) &&
         fits(value.expert_arena.activation_scale_bits, tokens * 32U) &&
         fits(value.expert_arena.gate_or_middle_bf16,
              tokens * 2048U * 2U) &&
         fits(value.expert_arena.up_bf16, tokens * 2048U * 2U) &&
         fits(value.expert_arena.error_flag_u32, sizeof(std::uint32_t));
}

}  // namespace

Result<DeepSeekDsparkMoeStageOperation>
DeepSeekDsparkMoeStageOperation::Create(
    DeepSeekDsparkStageId stage,
    DeepSeekDsparkMoeStageOperations& operations,
    std::uint32_t* host_error_flag) {
  if (!is_valid_deepseek_dspark_stage(stage) || host_error_flag == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek DSpark MoE stage identity is invalid");
  }
  const auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekDsparkMoeStageOperation result;
  result.stage_ = stage;
  result.operations_ = &operations;
  result.host_error_flag_ = host_error_flag;
  return result;
}

Status DeepSeekDsparkMoeStageOperation::poison(Status status) {
  state_ = State::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek DSpark MoE poisoned")
                     : status;
}

Status DeepSeekDsparkMoeStageOperation::launch(
    const DeepSeekPipelinePlanDescriptor& plan,
    const DeepSeekDsparkMtpStageResources& resources) {
  if (state_ != State::kIdle || operations_ == nullptr ||
      plan.engine_epoch == 0 || plan.plan_sequence == 0 ||
      plan.phase != DeepSeekPlanPhase::kDecode || plan.token_count != 1 ||
      plan.sequence_count != 1 || resources.stage != stage_ ||
      resources.transaction == nullptr ||
      resources.transaction->state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      resources.transaction->prepare_epoch() != resources.prepare_epoch ||
      resources.transaction->stream() != resources.stream ||
      resources.prepare_epoch == 0 || !valid_resources(resources)) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark MoE stage is not launchable");
  }

  const auto& mhc = resources.weights->mhc;
  const DeepSeekMhcPreLaunch pre{
      resources.attention_output_hc_bf16,
      mhc.feed_forward_fn_f32,
      mhc.feed_forward_scale_f32,
      mhc.feed_forward_base_f32,
      mhc.feed_forward_norm_bf16,
      resources.mhc_workspace.post_mix_f32,
      resources.mhc_workspace.residual_mix_f32,
      resources.mhc_workspace.layer_input_bf16,
      resources.error_flag_u32,
      resources.stream,
      kTokenCount,
      4096,
      kEpsilon,
      kEpsilon,
      kEpsilon,
      2.0F,
      20};
  const DeepSeekRouterBf16GemmLaunch router{
      resources.mhc_workspace.layer_input_bf16,
      resources.weights->router_weight_bf16,
      resources.router_scores_f32,
      resources.error_flag_u32,
      resources.stream,
      kTokenCount,
      256,
      4096};
  auto status = validate_deepseek_mhc_pre_launch(pre);
  if (!status.ok()) return status;
  status = validate_deepseek_router_bf16_gemm_launch(router);
  if (!status.ok()) return status;

  auto claimed = resources.transaction->claim_external_error_channel(
      host_error_flag_, resources.error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  active_ = &resources;
  const auto run = [this](Status value) {
    return value.ok() ? value : poison(value);
  };
  if (*claimed) {
    *host_error_flag_ = 0;
    status = run(operations_->zero_u32_async(
        resources.error_flag_u32, resources.stream));
    if (!status.ok()) return status;
  }
  status = run(operations_->zero_bytes_async(
      resources.expert_accumulator_f32, kAccumulatorBytes,
      resources.stream));
  if (!status.ok()) return status;
  status = run(operations_->mhc_pre(pre));
  if (!status.ok()) return status;
  status = run(operations_->router_gemm(router));
  if (!status.ok()) return status;
  status = run(operations_->copy_d2h_async(
      resources.router_host_scores.data(), resources.router_scores_f32,
      resources.router_host_scores.size_bytes(), resources.stream));
  if (!status.ok()) return status;
  status = run(operations_->copy_d2h_async(
      resources.router_host_bias.data(), resources.weights->router_bias_f32,
      resources.router_host_bias.size_bytes(), resources.stream));
  if (!status.ok()) return status;
  status = run(operations_->copy_d2h_async(
      host_error_flag_, resources.error_flag_u32,
      sizeof(std::uint32_t), resources.stream));
  if (!status.ok()) return status;
  status = run(operations_->record_event(
      resources.completion_event, resources.stream));
  if (!status.ok()) return status;
  state_ = State::kRouter;
  return Status::Ok();
}

Status DeepSeekDsparkMoeStageOperation::launch_prefill(
    const DeepSeekPipelinePlanDescriptor&,
    const DeepSeekDsparkMtpPrefillResources&) {
  return Status::FailedPrecondition(
      "DeepSeek DSpark MoE cannot execute prefill work");
}

Status DeepSeekDsparkMoeStageOperation::launch_finalize() {
  const auto& resources = *active_;
  const auto& arena = resources.expert_arena;
  const auto& shared = resources.weights->shared_expert;
  const auto& mhc = resources.weights->mhc;
  const DeepSeekFp8ActivationQuantLaunch input_quant{
      resources.mhc_workspace.layer_input_bf16,
      arena.activation_e4m3.address,
      arena.activation_scale_bits.address,
      resources.error_flag_u32,
      resources.stream,
      kTokenCount,
      4096};
  const DeepSeekFp4GemmLaunch w1{
      arena.activation_e4m3.address, arena.activation_scale_bits.address,
      shared.w1.packed.address, shared.w1.scales.address,
      arena.gate_or_middle_bf16.address, resources.error_flag_u32,
      resources.stream, kTokenCount, 2048, 4096};
  const DeepSeekFp4GemmLaunch w3{
      arena.activation_e4m3.address, arena.activation_scale_bits.address,
      shared.w3.packed.address, shared.w3.scales.address,
      arena.up_bf16.address, resources.error_flag_u32,
      resources.stream, kTokenCount, 2048, 4096};
  const DeepSeekSharedExpertSwiGluLaunch swiglu{
      arena.gate_or_middle_bf16.address, arena.up_bf16.address,
      arena.gate_or_middle_bf16.address, resources.error_flag_u32,
      resources.stream, kTokenCount};
  const DeepSeekFp8ActivationQuantLaunch middle_quant{
      arena.gate_or_middle_bf16.address, arena.activation_e4m3.address,
      arena.activation_scale_bits.address, resources.error_flag_u32,
      resources.stream, kTokenCount, 2048};
  const DeepSeekFp4GemmLaunch w2{
      arena.activation_e4m3.address, arena.activation_scale_bits.address,
      shared.w2.packed.address, shared.w2.scales.address,
      resources.mhc_workspace.ffn_branch_output_bf16,
      resources.error_flag_u32, resources.stream,
      kTokenCount, 4096, 2048};
  const DeepSeekExpertFinalizeLaunch finalize{
      resources.expert_accumulator_f32,
      resources.mhc_workspace.ffn_branch_output_bf16,
      arena.route_input_bf16.address,
      resources.error_flag_u32,
      resources.stream,
      kTokenCount};
  const DeepSeekMhcPostLaunch post{
      arena.route_input_bf16.address,
      resources.attention_output_hc_bf16,
      resources.mhc_workspace.post_mix_f32,
      resources.mhc_workspace.residual_mix_f32,
      resources.stage_output_hc_bf16,
      resources.error_flag_u32,
      resources.stream,
      kTokenCount,
      4096};

  for (const auto status : {
           validate_deepseek_fp8_activation_quant_launch(input_quant),
           validate_deepseek_fp4_gemm_launch(w1),
           validate_deepseek_fp4_gemm_launch(w3),
           validate_deepseek_shared_expert_swiglu_launch(swiglu),
           validate_deepseek_fp8_activation_quant_launch(middle_quant),
           validate_deepseek_fp4_gemm_launch(w2),
           validate_deepseek_mhc_post_launch(post)}) {
    if (!status.ok()) return poison(status);
  }
  const auto run = [this](Status value) {
    return value.ok() ? value : poison(value);
  };
  Status status = run(operations_->quant(input_quant));
  if (!status.ok()) return status;
  status = run(operations_->fp4_gemm(w1));
  if (!status.ok()) return status;
  status = run(operations_->fp4_gemm(w3));
  if (!status.ok()) return status;
  status = run(operations_->shared_swiglu(swiglu));
  if (!status.ok()) return status;
  status = run(operations_->quant(middle_quant));
  if (!status.ok()) return status;
  status = run(operations_->fp4_gemm(w2));
  if (!status.ok()) return status;
  status = run(operations_->finalize(finalize));
  if (!status.ok()) return status;
  status = run(operations_->mhc_post(post));
  if (!status.ok()) return status;
  status = run(operations_->copy_d2h_async(
      host_error_flag_, resources.error_flag_u32,
      sizeof(std::uint32_t), resources.stream));
  if (!status.ok()) return status;
  status = run(operations_->record_event(
      resources.completion_event, resources.stream));
  if (!status.ok()) return status;
  state_ = State::kFinalize;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus>
DeepSeekDsparkMoeStageOperation::poll() {
  if (state_ == State::kIdle || state_ == State::kPoisoned ||
      active_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark MoE stage is not pollable");
  }
  if (state_ == State::kRouter || state_ == State::kFinalize) {
    auto event = operations_->query_event(active_->completion_event);
    if (!event.ok()) return poison(event.status());
    if (*event == DeepSeekExpertAsyncStatus::kInProgress) {
      return DeepSeekStageComputeStatus::kInProgress;
    }
    if (*event == DeepSeekExpertAsyncStatus::kError ||
        *host_error_flag_ != 0) {
      (void)poison(Status::Internal("DeepSeek DSpark MoE device failed"));
      return DeepSeekStageComputeStatus::kError;
    }
    if (*event != DeepSeekExpertAsyncStatus::kSuccess) {
      return poison(Status::Internal(
          "DeepSeek DSpark MoE received invalid event state"));
    }
    if (state_ == State::kFinalize) {
      active_ = nullptr;
      route_plan_.reset();
      executor_.reset();
      state_ = State::kIdle;
      return DeepSeekStageComputeStatus::kSuccess;
    }

    std::vector<float> scores(active_->router_host_scores.begin(),
                              active_->router_host_scores.end());
    std::array<float, DeepSeekExpertSubwavePlan::kExpertCount> bias{};
    std::copy(active_->router_host_bias.begin(),
              active_->router_host_bias.end(), bias.begin());
    auto plan = DeepSeekLearnedRouter::Route(kTokenCount, scores, bias);
    if (!plan.ok()) return poison(plan.status());
    route_plan_.emplace(std::move(*plan));
    auto executor = DeepSeekDsparkResidentSubwaveExecutor::Create(
        stage_, *route_plan_, *active_->resident_experts);
    if (!executor.ok()) return poison(executor.status());
    executor_.emplace(std::move(*executor));
    state_ = State::kExperts;
    const auto status = executor_->advance(*active_->expert_kernel);
    if (!status.ok()) return poison(status);
    return DeepSeekStageComputeStatus::kInProgress;
  }

  const auto status = executor_->advance(*active_->expert_kernel);
  if (!status.ok()) return poison(status);
  if (executor_->state() == DeepSeekExpertSubwaveExecutorState::kPoisoned) {
    (void)poison(Status::Internal("DeepSeek DSpark routed experts failed"));
    return DeepSeekStageComputeStatus::kError;
  }
  if (executor_->state() !=
      DeepSeekExpertSubwaveExecutorState::kComplete) {
    return DeepSeekStageComputeStatus::kInProgress;
  }
  const auto finalized = launch_finalize();
  if (!finalized.ok()) return finalized;
  return DeepSeekStageComputeStatus::kInProgress;
}

Status DeepSeekDsparkMoeStageOperation::cancel() {
  Status status = Status::Ok();
  if (active_ != nullptr && operations_ != nullptr) {
    status = operations_->synchronize_stream(active_->stream);
  }
  active_ = nullptr;
  route_plan_.reset();
  executor_.reset();
  state_ = State::kPoisoned;
  return status;
}

}  // namespace pih
