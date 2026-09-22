#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct SparseAttentionLaunch final {
  // One sequence: BF16 query/output [tokens,heads,512], shared KV [kv_rows,512],
  // FP32 sink [heads], I32 indices [tokens,picks]. -1 is masked; all other
  // indices must address the caller-admitted concatenated window/compressed KV.
  EngramDeviceRegion query, kv, sink, indices, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, heads = 0, kv_rows = 0, picks = 0;
};
Status ValidateSparseAttention(const SparseAttentionLaunch& launch);
Status LaunchSparseAttention(const SparseAttentionLaunch& launch);
}  // namespace pih::deepseek_v41
