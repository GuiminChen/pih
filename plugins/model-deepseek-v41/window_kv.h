#pragma once
#include "norm_launch.h"
#include "rope_launch.h"

namespace pih::deepseek_v41 {
struct WindowKvLaunch final {
  // One sequence: mutable BF16 [tokens,512] and BF16 ring [128,512].
  // Quantize/dequantize the complete post-RoPE vector before seeding the ring.
  EngramDeviceRegion kv, ring, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t start = 0, tokens = 0;
};
Status ValidateWindowKv(const WindowKvLaunch& launch);
Status LaunchWindowKv(const WindowKvLaunch& launch);
struct WindowKvPrepareLaunch final {
  RmsNormLaunch norm;
  RopeApplyLaunch rope;
  WindowKvLaunch cache;
};
// Input is the BF16 KV projection. Phases must correspond to the step's query
// positions and layer; their semantic identity is admitted by the caller.
Status ValidateWindowKvPrepare(const WindowKvPrepareLaunch& launch);
Status LaunchWindowKvPrepare(const WindowKvPrepareLaunch& launch);
}  // namespace pih::deepseek_v41
