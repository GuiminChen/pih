#pragma once
#include "indexer_input.h"
#include "indexer_score.h"
#include "indexer_select.h"
#include "engram_completion.h"
#include "config.h"
#include <chrono>
#include <optional>

namespace pih::deepseek_v41 {
struct IndexerPipelineLaunch final {
  IndexerQueryLaunch query;
  // Present only when this index-key owner emits new compressed rows.
  std::optional<IndexerKeyLaunch> key;
  IndexerWeightedScoreLaunch scoring;
  IndexerSelectLaunch selection;
  // Present only at the candidate source; source selection remains unmasked.
  std::optional<IndexerCandidatesLaunch> candidates;
};
Status ValidateIndexerPipeline(const IndexerPipelineLaunch& launch);
Status ValidateIndexerLayer(const FlashConfig& config, std::uint32_t layer,
    const IndexerPipelineLaunch& launch);

class IndexerSingleRank final {
 public:
  using Clock = std::chrono::steady_clock;
  IndexerSingleRank(const IndexerSingleRank&) = delete;
  IndexerSingleRank& operator=(const IndexerSingleRank&) = delete;
  IndexerSingleRank(IndexerSingleRank&& other) noexcept;
  IndexerSingleRank& operator=(IndexerSingleRank&&) = delete;
  static Result<IndexerSingleRank> Start(const FlashConfig& config, std::uint32_t layer,
      const IndexerPipelineLaunch& launch,
      const EngramCompletionResources& resources, Clock::time_point deadline);
  // false=pending; true=complete with zero device error. Never resubmits.
  Result<bool> Poll();
 private:
  IndexerSingleRank() = default;
  std::optional<EngramCompletion> completion_;
  Clock::time_point deadline_{};
  bool failed_ = true, complete_ = false;
};
}  // namespace pih::deepseek_v41
