#pragma once
#include "rope_launch.h"

namespace pih::deepseek_v41 {
struct CompressedKvLaunch final {
  // Mutable post-RoPE BF16 [rows,512], and BF16 cache [capacity,512].
  // Simulate group-16 E2M1 with E4M3 scales, retaining dequantized BF16.
  EngramDeviceRegion kv, cache, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t rows = 0, first_slot = 0, capacity = 0;
};
Status ValidateCompressedKv(const CompressedKvLaunch& launch);
Status LaunchCompressedKv(const CompressedKvLaunch& launch);
struct CompressedKvPrepareLaunch final {
  // Preserve unrotated normalized latent for the indexer. Unlike standalone
  // RoPE, this chain requires distinct input/output storage.
  RopeApplyLaunch rope;
  CompressedKvLaunch cache;
};
Status ValidateCompressedKvPrepare(const CompressedKvPrepareLaunch& launch);
Status LaunchCompressedKvPrepare(const CompressedKvPrepareLaunch& launch);
}  // namespace pih::deepseek_v41
