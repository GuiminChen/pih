#include "indexer_score.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion x, std::uint64_t bytes, unsigned alignment) {
  return x.address && x.address % alignment == 0 && x.bytes == bytes &&
      bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Shape(unsigned tokens, unsigned heads) {
  return tokens && tokens <= 4096 && (heads == 4 || heads == 8 || heads == 16 || heads == 32);
}
template<std::size_t I, std::size_t W>
bool Disjoint(const std::array<EngramDeviceRegion, I>& reads, const std::array<EngramDeviceRegion, W>& writes) {
  for (std::size_t i = 0; i < W; ++i) {
    for (const auto read : reads) if (Overlap(read, writes[i])) return false;
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j])) return false;
  }
  return true;
}
}
Status ValidateIndexerWeights(const IndexerWeightsLaunch& x) {
  if (!x.stream || !Shape(x.tokens, x.heads) || !Valid(x.input, x.tokens * 5120ULL * 2, 2) ||
      !Valid(x.weight, x.heads * 5120ULL * 2, 2) || !Valid(x.output, x.tokens * std::uint64_t(x.heads) * 2, 2) ||
      !Valid(x.error_flag, 4, 4) || !Disjoint(std::array{x.input, x.weight}, std::array{x.output, x.error_flag}))
    return Status::InvalidArgument("V4.1 indexer head-weight layout or alias invalid");
  return Status::Ok();
}
Status ValidateIndexerScore(const IndexerScoreLaunch& x) {
  if (!x.stream || !Shape(x.tokens, x.heads) || !x.positions || x.positions > 1048576 ||
      !Valid(x.query, x.tokens * std::uint64_t(x.heads) * 128 * 2, 2) || !Valid(x.key, x.positions * 128ULL * 2, 2) ||
      !Valid(x.weights, x.tokens * std::uint64_t(x.heads) * 2, 2) ||
      !Valid(x.output, x.tokens * std::uint64_t(x.positions) * 2, 2) || !Valid(x.error_flag, 4, 4) ||
      !Disjoint(std::array{x.query, x.key, x.weights}, std::array{x.output, x.error_flag}))
    return Status::InvalidArgument("V4.1 indexer score layout or alias invalid");
  return Status::Ok();
}
Status ValidateIndexerWeightedScore(const IndexerWeightedScoreLaunch& x) {
  const auto weights = ValidateIndexerWeights(x.weights); if (!weights.ok()) return weights;
  const auto score = ValidateIndexerScore(x.score); if (!score.ok()) return score;
  if (x.weights.tokens != x.score.tokens || x.weights.heads != x.score.heads || x.weights.stream != x.score.stream ||
      !Same(x.weights.output, x.score.weights) || !Same(x.weights.error_flag, x.score.error_flag) ||
      !Disjoint(std::array{x.weights.input, x.weights.weight, x.score.query, x.score.key},
          std::array{x.weights.output, x.score.output, x.score.error_flag}))
    return Status::InvalidArgument("V4.1 weighted-score connection or alias invalid");
  return Status::Ok();
}
Status LaunchIndexerWeightedScore(const IndexerWeightedScoreLaunch& x) {
  const auto validation = ValidateIndexerWeightedScore(x); if (!validation.ok()) return validation;
  const auto weights = LaunchIndexerWeights(x.weights); if (!weights.ok()) return weights;
  return LaunchIndexerScore(x.score);
}
}  // namespace pih::deepseek_v41
