#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct IndexerSelectLaunch final {
  // BF16 globally reduced scores [tokens,positions]; I32 output
  // [tokens,min(512,positions)]. positions=(start+tokens)/ratio.
  // If positions=0 both regions must be absent; no device work is submitted.
  EngramDeviceRegion scores, output, error_flag;
  // Optional U8 [tokens,positions] candidate-source mask, ratio-one only.
  EngramDeviceRegion candidates;
  std::uintptr_t stream = 0;
  std::uint32_t start = 0, tokens = 0, ratio = 0, positions = 0, offset = 0;
};
Status ValidateIndexerSelect(const IndexerSelectLaunch& launch);
// Causal visibility, top-512 by score, then ascending original cache index.
// Score ties prefer lower positions; invisible fillers become -1.
// Optional candidate masking; no implicit TP reduction.
Status LaunchIndexerSelect(const IndexerSelectLaunch& launch);
struct IndexerCandidatesLaunch final {
  // Ratio-one source: globally reduced BF16 scores and U8 output mask,
  // both [tokens,positions], positions=start+tokens. Block size 8, top 2048.
  EngramDeviceRegion scores, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t start = 0, tokens = 0, positions = 0;
};
Status ValidateIndexerCandidates(const IndexerCandidatesLaunch& launch);
Status LaunchIndexerCandidates(const IndexerCandidatesLaunch& launch);
}  // namespace pih::deepseek_v41
