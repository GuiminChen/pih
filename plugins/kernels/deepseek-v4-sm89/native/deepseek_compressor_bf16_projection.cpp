#include "pih/backend/cuda/deepseek_compressor_bf16_projection.h"

namespace pih {

Status validate_deepseek_compressor_bf16_projection_launch(
    const DeepSeekCompressorBf16ProjectionLaunch& value) {
  const bool geometry =
      (value.ratio == 4 && (value.head_dim == 512 || value.head_dim == 128)) ||
      (value.ratio == 128 && value.head_dim == 512);
  if (value.input_bf16 == 0 || value.kv_weight_bf16 == 0 || value.gate_weight_bf16 == 0 ||
      value.kv_projection_f32 == 0 || value.gate_projection_f32 == 0 ||
      value.error_flag_u32 == 0 || value.stream == 0 ||
      value.token_count == 0 || value.token_count > 4096 || !geometry ||
      value.hidden_size != 4096 ||
      value.kv_projection_f32 == value.gate_projection_f32 ||
      value.kv_projection_f32 == value.input_bf16 ||
      value.gate_projection_f32 == value.input_bf16 ||
      value.kv_projection_f32 == value.kv_weight_bf16 ||
      value.gate_projection_f32 == value.kv_weight_bf16 ||
      value.kv_projection_f32 == value.gate_weight_bf16 ||
      value.gate_projection_f32 == value.gate_weight_bf16) {
    return Status::InvalidArgument(
        "DeepSeek BF16 compressor projection launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
