#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekIndexerProjectionLaunch final {
  std::uintptr_t qr_bf16 = 0;             // [T,1024]
  std::uintptr_t hidden_bf16 = 0;         // [T,4096]
  std::uintptr_t wq_b_e4m3 = 0;          // [8192,1024]
  std::uintptr_t wq_b_scale_bits = 0;    // [64,8] UE8M0
  std::uintptr_t qr_e4m3 = 0;           // [T,1024] scratch
  std::uintptr_t qr_scale_bits = 0;     // [T,8] scratch
  std::uintptr_t weights_proj_bf16 = 0;   // [64,4096]
  std::uintptr_t frequencies_f32 = 0;     // [table_positions,64]
  std::uintptr_t positions_u32 = 0;       // [T]
  std::uintptr_t query_bf16 = 0;          // [T,64,128]
  std::uintptr_t head_weight_f32 = 0;     // [T,64]
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  std::uint32_t table_position_count = 0;
};

Status validate_deepseek_indexer_projection_launch(
    const DeepSeekIndexerProjectionLaunch& launch);
Status launch_deepseek_indexer_projection(
    DeepSeekIndexerProjectionLaunch launch);

}  // namespace pih
