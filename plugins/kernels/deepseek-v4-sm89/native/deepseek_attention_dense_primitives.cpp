#include "pih/backend/cuda/deepseek_attention_dense_primitives.h"

#include <cmath>

namespace pih { namespace {
bool head_count_valid(std::uint32_t count) { return count == 1 || count == 64; }
}  // namespace pih::<anonymous>

Status validate_deepseek_head_rms_launch(const DeepSeekHeadRmsLaunch& v) {
  if (v.input_bf16 == 0 || v.output_bf16 == 0 || v.error_flag_u32 == 0 ||
      v.stream == 0 || v.token_count == 0 || v.token_count > 4096 ||
      !head_count_valid(v.head_count) || v.head_dimension != 512 ||
      !std::isfinite(v.epsilon) || v.epsilon != 1.0e-6F) {
    return Status::InvalidArgument("DeepSeek head RMS launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_rotary_launch(const DeepSeekRotaryLaunch& v) {
  if (v.input_bf16 == 0 || v.frequencies_f32 == 0 ||
      v.positions_u32 == 0 || v.table_position_count == 0 ||
      v.table_position_count > 1048576 ||
      v.error_flag_u32 == 0 || v.stream == 0 || v.token_count == 0 ||
      v.token_count > 4096 || !head_count_valid(v.head_count) ||
      v.head_dimension != 512 || v.rope_dimension != 64) {
    return Status::InvalidArgument("DeepSeek rotary launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_kv_fp8_simulate_launch(
    const DeepSeekKvFp8SimulateLaunch& v) {
  if (v.kv_bf16 == 0 || v.error_flag_u32 == 0 || v.stream == 0 ||
      v.token_count == 0 || v.token_count > 4096 ||
      v.vector_dimension != 512 || v.quantized_dimension != 448 ||
      v.group_size != 64) {
    return Status::InvalidArgument(
        "DeepSeek KV FP8 simulation launch is invalid");
  }
  return Status::Ok();
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status validate_deepseek_grouped_bf16_gemm_launch(
    const DeepSeekGroupedBf16GemmLaunch& v) {
  if (v.input_bf16 == 0 || v.weight_bf16 == 0 || v.output_bf16 == 0 ||
      v.error_flag_u32 == 0 || v.stream == 0 || v.token_count == 0 ||
      v.token_count > 4096 || v.group_count != 8 ||
      v.output_per_group != 1024 || v.input_per_group != 4096 ||
      v.output_bf16 == v.input_bf16 || v.output_bf16 == v.weight_bf16) {
    return Status::InvalidArgument(
        "DeepSeek grouped BF16 GEMM launch is invalid");
  }
  return Status::Ok();
}
#endif

Status validate_deepseek_grouped_fp8_gemm_launch(
    const DeepSeekGroupedFp8GemmLaunch& v) {
  if (v.input_bf16 == 0 || v.activation_e4m3 == 0 ||
      v.activation_scale_bits == 0 || v.weight_e4m3 == 0 ||
      v.weight_scale_bits == 0 || v.output_bf16 == 0 ||
      v.error_flag_u32 == 0 || v.stream == 0 || v.token_count == 0 ||
      v.token_count > 4096 || v.group_count != 8 ||
      v.output_per_group != 1024 || v.input_per_group != 4096 ||
      v.output_bf16 == v.input_bf16) {
    return Status::InvalidArgument(
        "DeepSeek grouped FP8 GEMM launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
