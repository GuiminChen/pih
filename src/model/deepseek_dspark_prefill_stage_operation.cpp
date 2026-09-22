#include "pih/model/deepseek_dspark_prefill_stage_operation.h"

namespace pih {
namespace {

struct PrefillLaunches final {
  DeepSeekFp8ActivationQuantLaunch quant;
  DeepSeekFp8GemmLaunch gemm;
  DeepSeekRmsNormLaunch rms;
  DeepSeekRotaryLaunch rotary;
  DeepSeekKvFp8SimulateLaunch simulate;
  DeepSeekDsparkRecentStoreLaunch store;
};

Result<PrefillLaunches> assemble_launches(
    const DeepSeekDsparkMtpPrefillResources& resources) {
  PrefillLaunches launches;
  launches.quant = {
      resources.main_normalized_bf16, resources.activation_e4m3,
      resources.activation_scale_ue8m0, resources.error_flag_u32,
      resources.stream, resources.token_count, 4096};
  launches.gemm = {
      resources.activation_e4m3, resources.activation_scale_ue8m0,
      resources.weights.wkv_fp8, resources.weights.wkv_scale_ue8m0,
      resources.kv_scratch_bf16, resources.error_flag_u32,
      resources.stream, resources.token_count, 512, 4096,
      DeepSeekFp8GemmOutputType::kBf16};
  launches.rms = {
      resources.kv_scratch_bf16, resources.weights.kv_norm_bf16,
      resources.kv_scratch_bf16, resources.error_flag_u32,
      resources.stream, resources.token_count, 512, 1.0e-6F};
  launches.rotary = {
      resources.kv_scratch_bf16, resources.rope_frequencies_f32,
      resources.error_flag_u32, resources.stream, resources.token_count,
      1, 512, 64, false, resources.positions_u32,
      resources.position_table_count};
  launches.simulate = {
      resources.kv_scratch_bf16, resources.error_flag_u32,
      resources.stream, resources.token_count, 512, 448, 64};
  launches.store = {
      resources.kv_scratch_bf16, resources.positions_u32,
      resources.recent_state.recent_bf16.address,
      resources.error_flag_u32, resources.stream, resources.token_count,
      512, 128, resources.position_table_count};
  auto status = validate_deepseek_fp8_activation_quant_launch(
      launches.quant);
  if (!status.ok()) return status;
  status = validate_deepseek_fp8_gemm_launch(launches.gemm);
  if (!status.ok()) return status;
  status = validate_deepseek_rms_norm_launch(launches.rms);
  if (!status.ok()) return status;
  status = validate_deepseek_rotary_launch(launches.rotary);
  if (!status.ok()) return status;
  status = validate_deepseek_kv_fp8_simulate_launch(launches.simulate);
  if (!status.ok()) return status;
  status = validate_deepseek_dspark_recent_store_launch(launches.store);
  if (!status.ok()) return status;
  return launches;
}

}  // namespace

Result<DeepSeekDsparkPrefillStageOperation>
DeepSeekDsparkPrefillStageOperation::Create(
    DeepSeekDsparkPrefillStageOperations& operations,
    std::uint32_t* host_error_flag) {
  if (host_error_flag == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill host error is null");
  }
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekDsparkPrefillStageOperation value;
  value.operations_ = &operations;
  value.host_error_flag_ = host_error_flag;
  return value;
}

Status DeepSeekDsparkPrefillStageOperation::launch(
    const DeepSeekPipelinePlanDescriptor&,
    const DeepSeekDsparkMtpStageResources&) {
  return Status::FailedPrecondition(
      "DeepSeek DSpark prefill operation cannot execute decode work");
}

Status DeepSeekDsparkPrefillStageOperation::launch_prefill(
    const DeepSeekPipelinePlanDescriptor& plan,
    const DeepSeekDsparkMtpPrefillResources& resources) {
  if (active_ || poisoned_ || operations_ == nullptr ||
      plan.engine_epoch == 0 || plan.plan_sequence == 0 ||
      plan.phase != DeepSeekPlanPhase::kPrefill ||
      plan.sequence_count != 1 || plan.token_count != resources.token_count ||
      resources.transaction == nullptr ||
      resources.transaction->state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      resources.transaction->stream() != resources.stream ||
      resources.transaction->prepare_epoch() != resources.prepare_epoch ||
      resources.completion_event == 0) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark prefill operation is not launchable");
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
#define PIH_DSPARK_PREFILL_RUN(call) \
  do {                                       \
    status = run(call);                      \
    if (!status.ok()) return status;         \
  } while (false)
  PIH_DSPARK_PREFILL_RUN(operations_->quant(launches->quant));
  PIH_DSPARK_PREFILL_RUN(operations_->gemm(launches->gemm));
  PIH_DSPARK_PREFILL_RUN(operations_->rms(launches->rms));
  PIH_DSPARK_PREFILL_RUN(operations_->rotary(launches->rotary));
  PIH_DSPARK_PREFILL_RUN(
      operations_->kv_fp8_simulate(launches->simulate));
  PIH_DSPARK_PREFILL_RUN(
      operations_->recent_store(launches->store));
  PIH_DSPARK_PREFILL_RUN(operations_->copy_error_d2h_async(
      host_error_flag_, resources.error_flag_u32, resources.stream));
  PIH_DSPARK_PREFILL_RUN(operations_->record_event(
      resources.completion_event, resources.stream));
#undef PIH_DSPARK_PREFILL_RUN
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus>
DeepSeekDsparkPrefillStageOperation::poll() {
  if (!active_ || poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark prefill operation is not pollable");
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
        "DeepSeek DSpark prefill received invalid event state");
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

Status DeepSeekDsparkPrefillStageOperation::cancel() {
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
