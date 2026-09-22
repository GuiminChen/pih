#include "pih/backend/cuda/deepseek_compressor_bf16_store.h"

#include <cmath>

namespace pih {

Status validate_deepseek_compressor_bf16_store_launch(
    const DeepSeekCompressorBf16StoreLaunch& launch) {
  if (launch.compressed_f32 == 0 || launch.rms_weight_bf16 == 0 ||
      launch.cos_sin_cache_f32 == 0 || launch.destination_bf16 == 0 ||
      launch.error_flag_u32 == 0 || launch.stream == 0 ||
      (launch.head_dim != 128 && launch.head_dim != 512) ||
      launch.rope_head_dim != 64 ||
      launch.rope_position >= 1048576 || !std::isfinite(launch.rms_epsilon) ||
      launch.rms_epsilon <= 0.0F) {
    return Status::InvalidArgument(
        "DeepSeek compressor BF16 store launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
