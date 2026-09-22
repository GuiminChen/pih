#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekCompressorPoolingLaunch final {
  std::uintptr_t kv_projection_f32 = 0;
  std::uintptr_t gate_projection_f32 = 0;
  std::uintptr_t ape_row_f32 = 0;
  std::uintptr_t kv_state_f32 = 0;
  std::uintptr_t score_state_f32 = 0;
  std::uintptr_t output_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t batch_count = 0;
  std::uint32_t ratio = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t absolute_position = 0;
};

Status validate_deepseek_compressor_pooling_launch(
    const DeepSeekCompressorPoolingLaunch& launch);
Status launch_deepseek_compressor_pooling(
    DeepSeekCompressorPoolingLaunch launch);

}  // namespace pih
