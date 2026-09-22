#include "pih/model/deepseek_attention_projection_coordinator.h"

namespace pih {

Status validate_deepseek_attention_projection_submission(
    const DeepSeekAttentionProjectionSubmission& s) {
  auto status = validate_deepseek_fp8_activation_quant_launch(s.input_quant);
  if (!status.ok()) return status;
  status = validate_deepseek_fp8_gemm_launch(s.wq_a); if (!status.ok()) return status;
  status = validate_deepseek_rms_norm_launch(s.q_norm); if (!status.ok()) return status;
  status = validate_deepseek_fp8_activation_quant_launch(s.q_quant); if (!status.ok()) return status;
  status = validate_deepseek_fp8_gemm_launch(s.wq_b); if (!status.ok()) return status;
  status = validate_deepseek_head_rms_launch(s.q_head_rms); if (!status.ok()) return status;
  status = validate_deepseek_rotary_launch(s.q_rope); if (!status.ok()) return status;
  status = validate_deepseek_fp8_gemm_launch(s.wkv); if (!status.ok()) return status;
  status = validate_deepseek_rms_norm_launch(s.kv_norm); if (!status.ok()) return status;
  status = validate_deepseek_rotary_launch(s.kv_rope); if (!status.ok()) return status;
  status = validate_deepseek_kv_fp8_simulate_launch(s.kv_simulate);
  if (!status.ok()) return status;

  const auto tokens = s.input_quant.token_count;
  const auto error = s.input_quant.error_flag;
  const auto stream = s.input_quant.stream;
  if (s.input_quant.logical_k != 4096 ||
      s.wq_a.m != tokens || s.wq_a.n != 1024 || s.wq_a.k != 4096 ||
      s.q_norm.rows != tokens || s.q_norm.hidden_size != 1024 ||
      s.q_quant.token_count != tokens || s.q_quant.logical_k != 1024 ||
      s.wq_b.m != tokens || s.wq_b.n != 32768 || s.wq_b.k != 1024 ||
      s.q_head_rms.token_count != tokens || s.q_head_rms.head_count != 64 ||
      s.q_rope.token_count != tokens || s.q_rope.head_count != 64 ||
      s.wkv.m != tokens || s.wkv.n != 512 || s.wkv.k != 4096 ||
      s.kv_norm.rows != tokens || s.kv_norm.hidden_size != 512 ||
      s.kv_rope.token_count != tokens || s.kv_rope.head_count != 1 ||
      s.kv_simulate.token_count != tokens) {
    return Status::InvalidArgument(
        "DeepSeek attention projection geometry is invalid");
  }
  if (s.wq_a.activation_e4m3 != s.input_quant.output_e4m3 ||
      s.wq_a.activation_scale_bits != s.input_quant.scale_bits ||
      s.q_norm.input_bf16 != s.wq_a.output_bf16 ||
      s.q_quant.input_bf16 != s.q_norm.output_bf16 ||
      s.wq_b.activation_e4m3 != s.q_quant.output_e4m3 ||
      s.wq_b.activation_scale_bits != s.q_quant.scale_bits ||
      s.q_head_rms.input_bf16 != s.wq_b.output_bf16 ||
      s.q_head_rms.output_bf16 != s.q_rope.input_bf16 ||
      s.wkv.activation_e4m3 != s.input_quant.output_e4m3 ||
      s.wkv.activation_scale_bits != s.input_quant.scale_bits ||
      s.kv_norm.input_bf16 != s.wkv.output_bf16 ||
      s.kv_norm.output_bf16 != s.kv_rope.input_bf16 ||
      s.kv_rope.input_bf16 != s.kv_simulate.kv_bf16 ||
      s.q_rope.frequencies_f32 != s.kv_rope.frequencies_f32 ||
      s.q_rope.positions_u32 != s.kv_rope.positions_u32 ||
      s.q_rope.table_position_count !=
          s.kv_rope.table_position_count) {
    return Status::InvalidArgument(
        "DeepSeek attention projection dataflow is invalid");
  }
  const std::uintptr_t errors[] = {
      s.wq_a.error_flag, s.q_norm.error_flag, s.q_quant.error_flag,
      s.wq_b.error_flag, s.q_head_rms.error_flag_u32,
      s.q_rope.error_flag_u32, s.wkv.error_flag, s.kv_norm.error_flag,
      s.kv_rope.error_flag_u32, s.kv_simulate.error_flag_u32};
  const std::uintptr_t streams[] = {
      s.wq_a.stream, s.q_norm.stream, s.q_quant.stream, s.wq_b.stream,
      s.q_head_rms.stream, s.q_rope.stream, s.wkv.stream, s.kv_norm.stream,
      s.kv_rope.stream, s.kv_simulate.stream};
  for (const auto value : errors) if (value != error)
    return Status::InvalidArgument("DeepSeek projection error channels differ");
  for (const auto value : streams) if (value != stream)
    return Status::InvalidArgument("DeepSeek projection streams differ");
  return Status::Ok();
}

Result<DeepSeekAttentionProjectionCoordinator>
DeepSeekAttentionProjectionCoordinator::Create(
    DeepSeekAttentionProjectionOperations& operations,
    std::uint32_t* host_error_flag) {
  if (host_error_flag == nullptr)
    return Status::InvalidArgument("DeepSeek projection host error is null");
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekAttentionProjectionCoordinator value;
  value.operations_ = &operations;
  value.host_error_flag_ = host_error_flag;
  return value;
}

Status DeepSeekAttentionProjectionCoordinator::run(Status status) {
  if (!status.ok()) poisoned_ = true;
  return status;
}

Status DeepSeekAttentionProjectionCoordinator::launch(
    const DeepSeekAttentionProjectionSubmission& s,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ || transaction.state() !=
                       DeepSeekAttentionSequenceTransactionState::kPreparing ||
      transaction.stream() != s.input_quant.stream) {
    return Status::FailedPrecondition(
        "DeepSeek projection coordinator is not launchable");
  }
  auto status = validate_deepseek_attention_projection_submission(s);
  if (!status.ok()) return status;
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, s.input_quant.error_flag);
  if (!claimed.ok()) return claimed.status();
  if (*claimed) {
    *host_error_flag_ = 0;
    status = run(operations_->zero_u32_async(
        s.input_quant.error_flag, s.input_quant.stream));
    if (!status.ok()) return status;
  }
#define PIH_RUN(call) do { status = run(call); if (!status.ok()) return status; } while (false)
  PIH_RUN(operations_->quant(s.input_quant));
  PIH_RUN(operations_->gemm(s.wq_a));
  PIH_RUN(operations_->rms(s.q_norm));
  PIH_RUN(operations_->quant(s.q_quant));
  PIH_RUN(operations_->gemm(s.wq_b));
  PIH_RUN(operations_->head_rms(s.q_head_rms));
  PIH_RUN(operations_->rotary(s.q_rope));
  PIH_RUN(operations_->gemm(s.wkv));
  PIH_RUN(operations_->rms(s.kv_norm));
  PIH_RUN(operations_->rotary(s.kv_rope));
  PIH_RUN(operations_->kv_simulate(s.kv_simulate));
  PIH_RUN(operations_->copy_error_d2h_async(
      host_error_flag_, s.input_quant.error_flag, s.input_quant.stream));
#undef PIH_RUN
  return Status::Ok();
}

}  // namespace pih
