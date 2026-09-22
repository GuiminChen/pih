#pragma once
#include "mhc_launch.h"

namespace pih::deepseek_v41 {
struct TokenEmbeddingLaunch final {
  // U32 token IDs; BF16 vocabulary shard [129280/world,5120], BF16 hidden
  // [tokens,5120], BF16 residual [tokens,4,5120], FP32 initial pre [tokens,4].
  EngramDeviceRegion ids, weight, hidden, residual, pre, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, world_size = 0, rank = 0;
};
Status ValidateTokenEmbedding(const TokenEmbeddingLaunch& launch);
Status LaunchTokenEmbeddingLookup(const TokenEmbeddingLaunch& launch);
// TP SUM of hidden must be enqueued before expansion on this same stream.
Status LaunchTokenEmbeddingExpand(const TokenEmbeddingLaunch& launch);
Status LaunchTokenEmbeddingSingleRank(const TokenEmbeddingLaunch& launch);
}  // namespace pih::deepseek_v41
