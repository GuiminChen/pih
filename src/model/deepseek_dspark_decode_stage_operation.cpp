#include "pih/model/deepseek_dspark_decode_stage_operation.h"

#include <algorithm>

namespace pih {
namespace {

constexpr std::uint32_t kDraftTokens =
    DeepSeekDsparkAttentionLaunch::kBlockSize;
constexpr float kEpsilon = 1.0e-6F;

struct DecodeLaunches final {
  DeepSeekDsparkPositionLaunch positions;
  DeepSeekMhcPreLaunch mhc_pre;
  DeepSeekFp8ActivationQuantLaunch main_quant;
  DeepSeekFp8GemmLaunch main_wkv;
  DeepSeekRmsNormLaunch main_kv_norm;
  DeepSeekRotaryLaunch main_kv_rope;
  DeepSeekKvFp8SimulateLaunch main_kv_simulate;
  DeepSeekDsparkRecentStoreLaunch main_recent_store;
  DeepSeekAttentionProjectionLayerSubmissions draft;
  DeepSeekDsparkAttentionLaunch attention;
  DeepSeekMhcPostLaunch mhc_post;
};

Result<DecodeLaunches> assemble_launches(
    const DeepSeekDsparkMtpStageResources& resources) {
  if (resources.weights == nullptr ||
      resources.weights->stage != resources.stage ||
      resources.weights->generation == 0 ||
      resources.weights->generation != resources.weight_generation ||
      resources.recent_state.stage != resources.stage ||
      resources.recent_state.recent_bf16.address == 0 ||
      resources.recent_state.recent_bf16.bytes !=
          kDeepSeekDsparkRecentStateBytes ||
      resources.stage_input_hc_bf16 == 0 ||
      resources.attention_output_hc_bf16 == 0 ||
      resources.stage_input_hc_bf16 ==
          resources.attention_output_hc_bf16 ||
      resources.main_normalized_bf16 == 0 ||
      resources.main_activation_e4m3 == 0 ||
      resources.main_activation_scale_ue8m0 == 0 ||
      resources.main_kv_bf16 == 0 || resources.main_positions_u32 == 0 ||
      resources.draft_positions_u32 == 0 ||
      resources.rope_frequencies_f32 == 0 ||
      resources.error_flag_u32 == 0 || resources.stream == 0 ||
      resources.completion_event == 0 ||
      resources.position_table_count == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode stage resources are incomplete");
  }

  DecodeLaunches result;
  result.positions = {
      resources.draft_positions_u32, resources.error_flag_u32,
      resources.stream, resources.current_position,
      resources.position_table_count};
  const auto& mhc = resources.weights->mhc;
  result.mhc_pre = {
      resources.stage_input_hc_bf16,
      mhc.attention_fn_f32,
      mhc.attention_scale_f32,
      mhc.attention_base_f32,
      mhc.attention_norm_bf16,
      resources.mhc_workspace.post_mix_f32,
      resources.mhc_workspace.residual_mix_f32,
      resources.mhc_workspace.layer_input_bf16,
      resources.error_flag_u32,
      resources.stream,
      kDraftTokens,
      4096,
      kEpsilon,
      kEpsilon,
      kEpsilon,
      2.0F,
      20};

  const auto& attention = resources.weights->attention;
  result.main_quant = {
      resources.main_normalized_bf16,
      resources.main_activation_e4m3,
      resources.main_activation_scale_ue8m0,
      resources.error_flag_u32,
      resources.stream,
      1,
      4096};
  result.main_wkv = {
      resources.main_activation_e4m3,
      resources.main_activation_scale_ue8m0,
      attention.wkv_fp8,
      attention.wkv_scale_ue8m0,
      resources.main_kv_bf16,
      resources.error_flag_u32,
      resources.stream,
      1,
      512,
      4096,
      DeepSeekFp8GemmOutputType::kBf16};
  result.main_kv_norm = {
      resources.main_kv_bf16, attention.kv_norm_bf16,
      resources.main_kv_bf16, resources.error_flag_u32,
      resources.stream, 1, 512, kEpsilon};
  result.main_kv_rope = {
      resources.main_kv_bf16, resources.rope_frequencies_f32,
      resources.error_flag_u32, resources.stream, 1, 1, 512, 64, false,
      resources.main_positions_u32, resources.position_table_count};
  result.main_kv_simulate = {
      resources.main_kv_bf16, resources.error_flag_u32,
      resources.stream, 1, 512, 448, 64};
  result.main_recent_store = {
      resources.main_kv_bf16, resources.main_positions_u32,
      resources.recent_state.recent_bf16.address,
      resources.error_flag_u32, resources.stream, 1, 512, 128,
      resources.position_table_count};

  auto draft_workspace = resources.attention_workspace;
  draft_workspace.positions_u32 = resources.draft_positions_u32;
  draft_workspace.error_flag_u32 = resources.error_flag_u32;
  auto draft = DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
      attention, draft_workspace, kDraftTokens, kDraftTokens,
      resources.mhc_workspace.layer_input_bf16,
      draft_workspace.attention_output_bf16,
      resources.rope_frequencies_f32, resources.position_table_count,
      resources.stream);
  if (!draft.ok()) return draft.status();
  result.draft = std::move(*draft);

  const auto recent_count = std::min<std::uint32_t>(
      DeepSeekDsparkAttentionLaunch::kWindowSize,
      resources.current_position + 1U);
  result.attention = {
      result.draft.sparse_query_bf16,
      resources.recent_state.recent_bf16.address,
      result.draft.sparse_kv_bf16,
      attention.attention_sink_f32,
      result.draft.sparse_output_bf16,
      resources.error_flag_u32,
      resources.stream,
      kDraftTokens,
      recent_count};
  result.mhc_post = {
      result.draft.branch_output_bf16,
      resources.stage_input_hc_bf16,
      resources.mhc_workspace.post_mix_f32,
      resources.mhc_workspace.residual_mix_f32,
      resources.attention_output_hc_bf16,
      resources.error_flag_u32,
      resources.stream,
      kDraftTokens,
      4096};

  for (const auto status : {
           validate_deepseek_dspark_position_launch(result.positions),
           validate_deepseek_mhc_pre_launch(result.mhc_pre),
           validate_deepseek_fp8_activation_quant_launch(result.main_quant),
           validate_deepseek_fp8_gemm_launch(result.main_wkv),
           validate_deepseek_rms_norm_launch(result.main_kv_norm),
           validate_deepseek_rotary_launch(result.main_kv_rope),
           validate_deepseek_kv_fp8_simulate_launch(
               result.main_kv_simulate),
           validate_deepseek_dspark_recent_store_launch(
               result.main_recent_store),
           validate_deepseek_dspark_attention_launch(result.attention),
           validate_deepseek_mhc_post_launch(result.mhc_post)}) {
    if (!status.ok()) return status;
  }
  return result;
}

}  // namespace

Result<DeepSeekDsparkDecodeStageOperation>
DeepSeekDsparkDecodeStageOperation::Create(
    DeepSeekDsparkStageId stage,
    DeepSeekDsparkDecodeStageOperations& operations,
    std::uint32_t* host_error_flag) {
  if (!is_valid_deepseek_dspark_stage(stage) || host_error_flag == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode stage identity is invalid");
  }
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekDsparkDecodeStageOperation value;
  value.stage_ = stage;
  value.operations_ = &operations;
  value.host_error_flag_ = host_error_flag;
  return value;
}

Status DeepSeekDsparkDecodeStageOperation::launch(
    const DeepSeekPipelinePlanDescriptor& plan,
    const DeepSeekDsparkMtpStageResources& resources) {
  if (active_ || poisoned_ || operations_ == nullptr ||
      plan.engine_epoch == 0 || plan.plan_sequence == 0 ||
      plan.phase != DeepSeekPlanPhase::kDecode || plan.token_count != 1 ||
      plan.sequence_count != 1 || resources.stage != stage_ ||
      resources.transaction == nullptr ||
      resources.transaction->state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      resources.transaction->stream() != resources.stream ||
      resources.transaction->prepare_epoch() != resources.prepare_epoch ||
      resources.prepare_epoch == 0) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark decode stage is not launchable");
  }
  auto launches = assemble_launches(resources);
  if (!launches.ok()) return launches.status();
  auto claimed = resources.transaction->claim_external_error_channel(
      host_error_flag_, resources.error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  active_ = true;
  active_stream_ = resources.stream;
  active_event_ = resources.completion_event;
  const auto run = [this](Status status) {
    if (!status.ok()) poisoned_ = true;
    return status;
  };
  Status status = Status::Ok();
  if (*claimed) {
    *host_error_flag_ = 0;
    status = run(operations_->zero_u32_async(
        resources.error_flag_u32, resources.stream));
    if (!status.ok()) return status;
  }
#define PIH_DSPARK_DECODE_RUN(call) \
  do {                                      \
    status = run(call);                     \
    if (!status.ok()) return status;        \
  } while (false)
  PIH_DSPARK_DECODE_RUN(operations_->positions(launches->positions));
  PIH_DSPARK_DECODE_RUN(operations_->mhc_pre(launches->mhc_pre));
  PIH_DSPARK_DECODE_RUN(operations_->quant(launches->main_quant));
  PIH_DSPARK_DECODE_RUN(operations_->gemm(launches->main_wkv));
  PIH_DSPARK_DECODE_RUN(operations_->rms(launches->main_kv_norm));
  PIH_DSPARK_DECODE_RUN(operations_->rotary(launches->main_kv_rope));
  PIH_DSPARK_DECODE_RUN(
      operations_->kv_simulate(launches->main_kv_simulate));
  PIH_DSPARK_DECODE_RUN(
      operations_->recent_store(launches->main_recent_store));
  const auto& input = launches->draft.input;
  PIH_DSPARK_DECODE_RUN(operations_->quant(input.input_quant));
  PIH_DSPARK_DECODE_RUN(operations_->gemm(input.wq_a));
  PIH_DSPARK_DECODE_RUN(operations_->rms(input.q_norm));
  PIH_DSPARK_DECODE_RUN(operations_->quant(input.q_quant));
  PIH_DSPARK_DECODE_RUN(operations_->gemm(input.wq_b));
  PIH_DSPARK_DECODE_RUN(operations_->head_rms(input.q_head_rms));
  PIH_DSPARK_DECODE_RUN(operations_->rotary(input.q_rope));
  PIH_DSPARK_DECODE_RUN(operations_->gemm(input.wkv));
  PIH_DSPARK_DECODE_RUN(operations_->rms(input.kv_norm));
  PIH_DSPARK_DECODE_RUN(operations_->rotary(input.kv_rope));
  PIH_DSPARK_DECODE_RUN(operations_->kv_simulate(input.kv_simulate));
  PIH_DSPARK_DECODE_RUN(operations_->attention(launches->attention));
  const auto& output = launches->draft.output;
  PIH_DSPARK_DECODE_RUN(operations_->rotary(output.inverse_rope));
  PIH_DSPARK_DECODE_RUN(operations_->grouped_gemm(output.wo_a));
  PIH_DSPARK_DECODE_RUN(operations_->quant(output.quant));
  PIH_DSPARK_DECODE_RUN(operations_->gemm(output.wo_b));
  PIH_DSPARK_DECODE_RUN(operations_->mhc_post(launches->mhc_post));
  PIH_DSPARK_DECODE_RUN(operations_->copy_error_d2h_async(
      host_error_flag_, resources.error_flag_u32, resources.stream));
  PIH_DSPARK_DECODE_RUN(operations_->record_event(
      resources.completion_event, resources.stream));
#undef PIH_DSPARK_DECODE_RUN
  return Status::Ok();
}

Status DeepSeekDsparkDecodeStageOperation::launch_prefill(
    const DeepSeekPipelinePlanDescriptor&,
    const DeepSeekDsparkMtpPrefillResources&) {
  return Status::FailedPrecondition(
      "DeepSeek DSpark decode operation cannot execute prefill work");
}

Result<DeepSeekStageComputeStatus>
DeepSeekDsparkDecodeStageOperation::poll() {
  if (!active_ || poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark decode stage is not pollable");
  }
  auto event = operations_->query_event(active_event_);
  if (!event.ok()) {
    poisoned_ = true;
    active_ = false;
    return event.status();
  }
  if (*event == DeepSeekExpertAsyncStatus::kInProgress) {
    return DeepSeekStageComputeStatus::kInProgress;
  }
  if (*event != DeepSeekExpertAsyncStatus::kSuccess &&
      *event != DeepSeekExpertAsyncStatus::kError) {
    poisoned_ = true;
    return Status::Internal(
        "DeepSeek DSpark decode received invalid event state");
  }
  active_ = false;
  active_stream_ = 0;
  active_event_ = 0;
  if (*event == DeepSeekExpertAsyncStatus::kError ||
      *host_error_flag_ != 0) {
    poisoned_ = true;
    return DeepSeekStageComputeStatus::kError;
  }
  return DeepSeekStageComputeStatus::kSuccess;
}

Status DeepSeekDsparkDecodeStageOperation::cancel() {
  Status status = Status::Ok();
  if (active_stream_ != 0 && operations_ != nullptr) {
    status = operations_->synchronize_stream(active_stream_);
  }
  active_ = false;
  active_stream_ = 0;
  active_event_ = 0;
  poisoned_ = true;
  return status;
}

}  // namespace pih
