#pragma once
#include "attention_sources.h"
#include "indexer_pipeline.h"

namespace pih::deepseek_v41 {
struct IndexedSourcesLaunch final {
  AttentionSourcesLaunch sources;
  // Present only for index-owner layers with a nonempty compressed prefix.
  std::optional<IndexerPipelineLaunch> indexer;
};
Status ValidateIndexedSources(const FlashConfig& config, const IndexedSourcesLaunch& launch);
class IndexedSourcesOperation final {
 public:
  using Clock = std::chrono::steady_clock;
  IndexedSourcesOperation(const IndexedSourcesOperation&) = delete;
  IndexedSourcesOperation& operator=(const IndexedSourcesOperation&) = delete;
  IndexedSourcesOperation(IndexedSourcesOperation&& other) noexcept;
  IndexedSourcesOperation& operator=(IndexedSourcesOperation&&) = delete;
  static Result<IndexedSourcesOperation> Start(const FlashConfig& config, const IndexedSourcesLaunch& launch,
      std::uint32_t rank, std::uintptr_t communicator, const EngramCompletionResources& resources, Clock::time_point deadline);
  Result<bool> Poll();
 private:
  IndexedSourcesOperation() = default;
  std::optional<IndexerTensorParallel> indexer_;
  std::optional<EngramCompletion> completion_;
  Clock::time_point deadline_{};
  bool failed_ = true, complete_ = false;
};
}  // namespace pih::deepseek_v41
