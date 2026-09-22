#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekDsparkRecentStoreLaunch final {
  std::uintptr_t source_bf16 = 0;
  std::uintptr_t positions_u32 = 0;
  std::uintptr_t recent_ring_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  std::uint32_t vector_dimension = 0;
  std::uint32_t ring_rows = 0;
  std::uint32_t maximum_position_count = 0;
};

Status validate_deepseek_dspark_recent_store_launch(
    const DeepSeekDsparkRecentStoreLaunch& launch);
Status launch_deepseek_dspark_recent_store(
    DeepSeekDsparkRecentStoreLaunch launch);

}  // namespace pih
