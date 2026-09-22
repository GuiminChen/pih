#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct IndexerWeightsLaunch final {
  // BF16 hidden [tokens,5120], BF16 weight [heads,5120], BF16 output
  // [tokens,heads]. Round projection to BF16, multiply by 1/64, round again.
  EngramDeviceRegion input, weight, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, heads = 0;
};
struct IndexerScoreLaunch final {
  // BF16 query [tokens,heads,128], key [positions,128], weights [tokens,heads],
  // output [tokens,positions]. Output is rank-local, BEFORE TP SUM or masking.
  EngramDeviceRegion query, key, weights, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0, heads = 0, positions = 0;
};
Status ValidateIndexerWeights(const IndexerWeightsLaunch& launch);
Status LaunchIndexerWeights(const IndexerWeightsLaunch& launch);
Status ValidateIndexerScore(const IndexerScoreLaunch& launch);
Status LaunchIndexerScore(const IndexerScoreLaunch& launch);
struct IndexerWeightedScoreLaunch final {
  IndexerWeightsLaunch weights;
  IndexerScoreLaunch score;
};
Status ValidateIndexerWeightedScore(const IndexerWeightedScoreLaunch& launch);
Status LaunchIndexerWeightedScore(const IndexerWeightedScoreLaunch& launch);
}  // namespace pih::deepseek_v41
