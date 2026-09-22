#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekCompressorBf16StoreLaunch final {
  std::uintptr_t compressed_f32 = 0;
  std::uintptr_t rms_weight_bf16 = 0;
  std::uintptr_t cos_sin_cache_f32 = 0;
  std::uintptr_t destination_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t rope_head_dim = 0;
  std::uint32_t rope_position = 0;
  float rms_epsilon = 0.0F;
};

Status validate_deepseek_compressor_bf16_store_launch(
    const DeepSeekCompressorBf16StoreLaunch& launch);
Status launch_deepseek_compressor_bf16_store(
    DeepSeekCompressorBf16StoreLaunch launch);

}  // namespace pih
