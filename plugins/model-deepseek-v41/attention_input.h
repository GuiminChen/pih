#pragma once
#include "linear_fp8.h"
#include "window_kv.h"

namespace pih::deepseek_v41 {
struct AttentionQueryLaunch final {
  Fp8LinearLaunch low_rank;  // 5120 -> 1280, replicated
  RmsNormLaunch norm;       // leaves normalized qr available for the indexer
  Fp8LinearLaunch expand;   // 1280 -> local_heads * 512
  RopeApplyLaunch rope;     // forward, original query positions
};
struct AttentionWindowLaunch final {
  Fp8LinearLaunch projection;  // 5120 -> 512, replicated
  WindowKvPrepareLaunch prepare;
};
Status ValidateAttentionQuery(const AttentionQueryLaunch& launch);
Status LaunchAttentionQuery(const AttentionQueryLaunch& launch);
Status ValidateAttentionWindow(const AttentionWindowLaunch& launch);
Status LaunchAttentionWindow(const AttentionWindowLaunch& launch);
}  // namespace pih::deepseek_v41
