#pragma once
#include "attention_assemble.h"
#include "mhc_launch.h"

namespace pih::deepseek_v41 {
struct AttentionResidualLaunch final {
  AssembledAttentionOutputLaunch attention;
  MhcPostLaunch residual;
};
Status ValidateAttentionResidual(const AttentionResidualLaunch& launch);
// Only eight local groups (TP=1). Multi-rank use requires a SUM between
// attention output and mHC and must not call this entry.
Status LaunchAttentionResidualSingleRank(const AttentionResidualLaunch& launch);
}  // namespace pih::deepseek_v41
