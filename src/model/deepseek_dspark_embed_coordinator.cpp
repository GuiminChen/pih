#include "pih/model/deepseek_dspark_embed_coordinator.h"

namespace pih { namespace {
Status validate_main_flow(const DeepSeekDsparkEmbedSubmission& s) {
  auto status = validate_deepseek_fp8_activation_quant_launch(s.main_quant);
  if (!status.ok()) return status;
  status = validate_deepseek_fp8_gemm_launch(s.main_proj);
  if (!status.ok()) return status;
  status = validate_deepseek_rms_norm_launch(s.main_norm);
  if (!status.ok()) return status;
  const auto rows = s.main_quant.token_count;
  const auto error = s.main_quant.error_flag;
  const auto stream = s.main_quant.stream;
  if (s.main_quant.logical_k != 12288 || s.main_proj.m != rows ||
      s.main_proj.n != 4096 || s.main_proj.k != 12288 ||
      s.main_norm.rows != rows || s.main_norm.hidden_size != 4096 ||
      s.main_proj.activation_e4m3 != s.main_quant.output_e4m3 ||
      s.main_proj.activation_scale_bits != s.main_quant.scale_bits ||
      s.main_norm.input_bf16 != s.main_proj.output_bf16 ||
      s.main_proj.error_flag != error || s.main_norm.error_flag != error ||
      s.main_proj.stream != stream || s.main_norm.stream != stream) {
    return Status::InvalidArgument(
        "DeepSeek DSpark embed dataflow is invalid");
  }
  return Status::Ok();
}

Status validate_decode_flow(const DeepSeekDsparkEmbedSubmission& s) {
  auto status = validate_main_flow(s);
  if (!status.ok()) return status;
  status = validate_deepseek_dspark_draft_init_launch(s.draft_init);
  if (!status.ok()) return status;
  if (s.draft_init.sequence_count != s.main_quant.token_count ||
      s.draft_init.error_flag_u32 != s.main_quant.error_flag ||
      s.draft_init.stream != s.main_quant.stream) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode embed dataflow is invalid");
  }
  return Status::Ok();
}
}  // namespace pih::<anonymous>
Result<DeepSeekDsparkEmbedCoordinator>
DeepSeekDsparkEmbedCoordinator::Create(
    DeepSeekDsparkEmbedOperations& operations,
    std::uint32_t* host_error_flag) {
  if (host_error_flag == nullptr)
    return Status::InvalidArgument("DeepSeek DSpark embed host error is null");
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekDsparkEmbedCoordinator value;
  value.operations_ = &operations;
  value.host_error_flag_ = host_error_flag;
  return value;
}
Status DeepSeekDsparkEmbedCoordinator::run(Status status) {
  if (!status.ok()) poisoned_ = true;
  return status;
}
Status DeepSeekDsparkEmbedCoordinator::launch(
    const DeepSeekDsparkEmbedSubmission& s,
    DeepSeekAttentionSequenceTransaction& transaction) {
  return launch_internal(s, transaction, true);
}

Status DeepSeekDsparkEmbedCoordinator::launch_prefill_state(
    const DeepSeekDsparkEmbedSubmission& s,
    DeepSeekAttentionSequenceTransaction& transaction) {
  return launch_internal(s, transaction, false);
}

Status DeepSeekDsparkEmbedCoordinator::launch_internal(
    const DeepSeekDsparkEmbedSubmission& s,
    DeepSeekAttentionSequenceTransaction& transaction,
    bool initialize_draft) {
  if (poisoned_ || transaction.state() !=
                       DeepSeekAttentionSequenceTransactionState::kPreparing ||
      transaction.stream() != s.main_quant.stream) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark embed coordinator is not launchable");
  }
  auto status = initialize_draft ? validate_decode_flow(s)
                                 : validate_main_flow(s);
  if (!status.ok()) return status;
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, s.main_quant.error_flag);
  if (!claimed.ok()) return claimed.status();
  if (*claimed) {
    *host_error_flag_ = 0;
    status = run(operations_->zero_u32_async(
        s.main_quant.error_flag, s.main_quant.stream));
    if (!status.ok()) return status;
  }
#define PIH_RUN(call) do { status = run(call); if (!status.ok()) return status; } while (false)
  PIH_RUN(operations_->quant(s.main_quant));
  PIH_RUN(operations_->gemm(s.main_proj));
  PIH_RUN(operations_->rms(s.main_norm));
  if (initialize_draft) {
    PIH_RUN(operations_->draft_init(s.draft_init));
  }
  PIH_RUN(operations_->copy_error_d2h_async(
      host_error_flag_, s.main_quant.error_flag, s.main_quant.stream));
#undef PIH_RUN
  return Status::Ok();
}
}  // namespace pih
