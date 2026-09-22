#include "indexer_reduce.h"

namespace pih::deepseek_v41 {
Status ValidateIndexerReduction(const IndexerScoreLaunch& x, std::uint32_t rank, std::uintptr_t communicator) {
  const auto layout = ValidateIndexerScore(x); if (!layout.ok()) return layout;
  return ValidateBf16Reduction({x.output, x.stream, 32 / x.heads, rank}, communicator);
}
Result<EngramReduction> ReduceIndexerScore(const IndexerScoreLaunch& x, std::uint32_t rank, std::uintptr_t communicator) {
  const auto layout = ValidateIndexerScore(x); if (!layout.ok()) return layout;
  return EngramReduction::SubmitBf16({x.output, x.stream, 32 / x.heads, rank}, communicator);
}
}  // namespace pih::deepseek_v41
