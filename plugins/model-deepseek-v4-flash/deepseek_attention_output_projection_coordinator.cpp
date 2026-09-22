#include "pih/model/deepseek_attention_output_projection_coordinator.h"

namespace pih {
Status validate_deepseek_attention_output_projection_submission(
    const DeepSeekAttentionOutputProjectionSubmission& s) {
  auto status = validate_deepseek_rotary_launch(s.inverse_rope);
  if (!status.ok()) return status;
  status = validate_deepseek_grouped_fp8_gemm_launch(s.wo_a);
  if (!status.ok()) return status;
  status = validate_deepseek_fp8_activation_quant_launch(s.quant);
  if (!status.ok()) return status;
  status = validate_deepseek_fp8_gemm_launch(s.wo_b);
  if (!status.ok()) return status;
  const auto tokens = s.inverse_rope.token_count;
  const auto error = s.inverse_rope.error_flag_u32;
  const auto stream = s.inverse_rope.stream;
  if (!s.inverse_rope.inverse || s.wo_a.token_count != tokens ||
      s.quant.token_count != tokens || s.quant.logical_k != 8192 ||
      s.wo_b.m != tokens || s.wo_b.n != 4096 || s.wo_b.k != 8192 ||
      s.wo_a.input_bf16 != s.inverse_rope.input_bf16 ||
      s.quant.input_bf16 != s.wo_a.output_bf16 ||
      s.wo_b.activation_e4m3 != s.quant.output_e4m3 ||
      s.wo_b.activation_scale_bits != s.quant.scale_bits ||
      s.wo_a.error_flag_u32 != error || s.quant.error_flag != error ||
      s.wo_b.error_flag != error || s.wo_a.stream != stream ||
      s.quant.stream != stream || s.wo_b.stream != stream) {
    return Status::InvalidArgument(
        "DeepSeek attention output projection flow is invalid");
  }
  return Status::Ok();
}
Result<DeepSeekAttentionOutputProjectionCoordinator>
DeepSeekAttentionOutputProjectionCoordinator::Create(
    DeepSeekAttentionOutputProjectionOperations& operations,
    std::uint32_t* host_error_flag) {
  if (host_error_flag == nullptr)
    return Status::InvalidArgument("DeepSeek output projection host error is null");
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekAttentionOutputProjectionCoordinator value;
  value.operations_ = &operations;
  value.host_error_flag_ = host_error_flag;
  return value;
}
Status DeepSeekAttentionOutputProjectionCoordinator::run(Status status) {
  if (!status.ok()) poisoned_ = true;
  return status;
}
Status DeepSeekAttentionOutputProjectionCoordinator::launch(
    const DeepSeekAttentionOutputProjectionSubmission& s,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ || transaction.state() !=
                       DeepSeekAttentionSequenceTransactionState::kPreparing ||
      transaction.stream() != s.inverse_rope.stream) {
    return Status::FailedPrecondition(
        "DeepSeek output projection coordinator is not launchable");
  }
  auto status = validate_deepseek_attention_output_projection_submission(s);
  if (!status.ok()) return status;
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, s.inverse_rope.error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  if (*claimed) {
    *host_error_flag_ = 0;
    status = run(operations_->zero_u32_async(
        s.inverse_rope.error_flag_u32, s.inverse_rope.stream));
    if (!status.ok()) return status;
  }
#define PIH_RUN(call) do { status = run(call); if (!status.ok()) return status; } while (false)
  PIH_RUN(operations_->rotary(s.inverse_rope));
  PIH_RUN(operations_->grouped_gemm(s.wo_a));
  PIH_RUN(operations_->quant(s.quant));
  PIH_RUN(operations_->gemm(s.wo_b));
  PIH_RUN(operations_->copy_error_d2h_async(
      host_error_flag_, s.inverse_rope.error_flag_u32,
      s.inverse_rope.stream));
#undef PIH_RUN
  return Status::Ok();
}
}  // namespace pih
