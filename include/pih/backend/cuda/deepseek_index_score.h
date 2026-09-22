#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekIndexScoreLaunch final {
  static constexpr std::uint32_t kHeadDim = 128;
  static constexpr std::uint32_t kMaximumHeads = 64;
  static constexpr std::uint32_t kMaximumSlotTile = 4096;

  std::uintptr_t query_bf16 = 0;       // [query_count, head_count, 128]
  std::uintptr_t index_kv_bf16 = 0;    // [slot_count, 128]
  std::uintptr_t head_weight_f32 = 0;  // [query_count, head_count], pre-scaled
  std::uintptr_t score_f32 = 0;        // [query_count, slot_count]
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t query_count = 0;
  std::uint32_t head_count = 0;        // local heads; TP contributions reduce later
  std::uint32_t slot_count = 0;        // one bounded history tile
  std::uintptr_t page_slots_u32 = 0;   // [logical_page_count], optional legacy=0
  std::uint32_t slot_base = 0;         // logical slot base of this tile
  std::uint32_t logical_page_count = 0;
  std::uint32_t physical_page_count = 0;
};

Status validate_deepseek_index_score_launch(
    const DeepSeekIndexScoreLaunch& launch);
Status launch_deepseek_index_score(DeepSeekIndexScoreLaunch launch);

}  // namespace pih
