#pragma once
#include "engram_reduce.h"
#include "indexer_score.h"

namespace pih::deepseek_v41 {
Status ValidateIndexerReduction(const IndexerScoreLaunch& score, std::uint32_t rank, std::uintptr_t communicator);
// Submit after score enqueue; wait for kEnqueued before selection on the same
// stream. Actual completion/error admission is a separate caller obligation.
Result<EngramReduction> ReduceIndexerScore(const IndexerScoreLaunch& score, std::uint32_t rank, std::uintptr_t communicator);
}  // namespace pih::deepseek_v41
