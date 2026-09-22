#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

// D-Spark decode attends every proposal query to the valid 128-row main-token
// window and to all five proposal-token KV rows.  This is deliberately a
// separate ABI from the main model's compressed/indexed sparse attention.
struct DeepSeekDsparkAttentionLaunch final {
  static constexpr std::uint32_t kBlockSize = 5;
  static constexpr std::uint32_t kHeadCount = 64;
  static constexpr std::uint32_t kHeadDimension = 512;
  static constexpr std::uint32_t kWindowSize = 128;

  std::uintptr_t query_bf16 = 0;
  std::uintptr_t recent_kv_bf16 = 0;
  std::uintptr_t draft_kv_bf16 = 0;
  std::uintptr_t attention_sink_f32 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t query_count = 0;
  std::uint32_t recent_count = 0;
};

// Builds the five proposal positions current_position + [1, 5] on device.
// The current main-token position itself remains in the request input table.
struct DeepSeekDsparkPositionLaunch final {
  std::uintptr_t draft_positions_u32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t current_position = 0;
  std::uint32_t table_position_count = 0;
};

Status validate_deepseek_dspark_attention_launch(
    const DeepSeekDsparkAttentionLaunch& launch);
Status launch_deepseek_dspark_attention(
    DeepSeekDsparkAttentionLaunch launch);
Status validate_deepseek_dspark_position_launch(
    const DeepSeekDsparkPositionLaunch& launch);
Status launch_deepseek_dspark_positions(
    DeepSeekDsparkPositionLaunch launch);

}  // namespace pih
