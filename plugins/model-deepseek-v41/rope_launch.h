#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct RopeTableLaunch final {
  // U32 original positions [tokens], FP32 [tokens,32,2] cosine/sine output.
  // Supply the first original position for a compressed token, not its cache index.
  EngramDeviceRegion positions, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, layer = 0;
};
struct RopeApplyLaunch final {
  // BF16 [tokens,heads,width], width=128 (indexer) or 512 (attention).
  // Only the last 64 elements rotate, in adjacent complex pairs. The prefix
  // is copied unchanged. Exact in-place input/output is supported, partial alias not.
  EngramDeviceRegion input, phases, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, heads = 0, width = 0;
  bool inverse = false;
};
Status ValidateRopeTable(const RopeTableLaunch& launch);
Status ValidateRopeApply(const RopeApplyLaunch& launch);
Status LaunchRopeTable(const RopeTableLaunch& launch);
Status LaunchRopeApply(const RopeApplyLaunch& launch);
struct RopeSequenceLaunch final {
  RopeTableLaunch table;
  std::uint32_t first_position = 0, stride = 0;
};
// Generates original U32 positions on-device, then their RoPE phase table.
Status ValidateRopeSequence(const RopeSequenceLaunch& launch);
Status LaunchRopeSequence(const RopeSequenceLaunch& launch);
}  // namespace pih::deepseek_v41
