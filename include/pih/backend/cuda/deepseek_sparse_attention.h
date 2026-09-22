#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekSparseAttentionLaunch final {
  static constexpr std::uint32_t kHeadDim = 512;
  static constexpr std::uint32_t kTileIndices = 64;

  std::uintptr_t query_bf16 = 0;
  std::uintptr_t latent_kv_bf16 = 0;
  std::uintptr_t attention_sink_f32 = 0;
  std::uintptr_t indices_i32 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t query_count = 0;
  std::uint32_t head_count = 0;
  std::uint32_t kv_count = 0;
  std::uint32_t index_count = 0;
  // Optional paged mode. Indices remain in the canonical logical address
  // space; recent and compressed ranges are translated to disjoint arenas.
  std::uintptr_t compressed_kv_bf16 = 0;
  std::uintptr_t page_slots_u32 = 0;
  std::int32_t recent_physical_offset = 0;
  std::int32_t compressed_physical_offset = 0;
  std::uint32_t compressed_slot_count = 0;
  std::uint32_t logical_page_count = 0;
  std::uint32_t physical_page_count = 0;
};

Status validate_deepseek_sparse_attention_launch(
    const DeepSeekSparseAttentionLaunch& launch);
Status launch_deepseek_sparse_attention(
    DeepSeekSparseAttentionLaunch launch);

}  // namespace pih
