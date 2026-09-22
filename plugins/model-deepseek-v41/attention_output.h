#pragma once
#include "sparse_attention.h"
#include "rope_launch.h"
#include "linear_fp8.h"

namespace pih::deepseek_v41 {
struct GroupedOutputLaunch final {
  // BF16 input [tokens,groups,4096], weight [groups,1024,4096],
  // output [tokens,groups,1024]. Each group contains eight 512-wide heads.
  EngramDeviceRegion input, weight, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, groups = 0;
};
Status ValidateGroupedOutput(const GroupedOutputLaunch& launch);
Status LaunchGroupedOutput(const GroupedOutputLaunch& launch);
struct AttentionOutputLaunch final {
  SparseAttentionLaunch attention;
  RopeApplyLaunch inverse_rope;
  GroupedOutputLaunch projection;
};
// Computes attention -> inverse tail RoPE -> wo_a. The FP8 wo_b projection and
// TP SUM remain separate required stages; this is not final hidden-state output.
Status ValidateAttentionOutput(const AttentionOutputLaunch& launch);
Status LaunchAttentionOutput(const AttentionOutputLaunch& launch);
struct AttentionLocalOutputLaunch final {
  AttentionOutputLaunch grouped;
  Fp8LinearLaunch linear;
  // Separate FP32 [tokens,5120] transport scratch. Never SUM BF16 directly:
  // reference wo_b rounds locally, promotes, reduces FP32, then rounds again.
  EngramDeviceRegion reduction;
};
// Includes FP8 wo_b. Output [tokens,5120] is a rank-local contribution and must
// be SUM-reduced when the eight global groups are partitioned across ranks.
Status ValidateAttentionLocalOutput(const AttentionLocalOutputLaunch& launch);
Status LaunchAttentionLocalOutput(const AttentionLocalOutputLaunch& launch);
Status PromoteAttentionOutput(const AttentionLocalOutputLaunch& launch);
// Only after FP32 SUM is enqueued on the same stream (or TP1 promotion).
Status RoundAttentionOutput(const AttentionLocalOutputLaunch& launch);
}  // namespace pih::deepseek_v41
