#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekCompressorBf16ProjectionLaunch final {
  std::uintptr_t input_bf16 = 0;
  std::uintptr_t kv_weight_bf16 = 0;
  std::uintptr_t gate_weight_bf16 = 0;
  std::uintptr_t kv_projection_f32 = 0;
  std::uintptr_t gate_projection_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  std::uint32_t ratio = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t hidden_size = 0;
};

Status validate_deepseek_compressor_bf16_projection_launch(
    const DeepSeekCompressorBf16ProjectionLaunch& launch);
Status launch_deepseek_compressor_bf16_projection(
    DeepSeekCompressorBf16ProjectionLaunch launch);

}  // namespace pih
