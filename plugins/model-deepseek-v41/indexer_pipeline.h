#pragma once
#include "indexer_reduce.h"
#include "indexer_chain.h"
#include "engram_completion.h"
#include <chrono>
#include <optional>

namespace pih::deepseek_v41 {
enum class IndexerPipelineState { kWaitingReduction, kWaitingCompletion, kComplete, kFailed };
class IndexerTensorParallel final {
 public:
  using Clock = std::chrono::steady_clock;
  IndexerTensorParallel(const IndexerTensorParallel&) = delete;
  IndexerTensorParallel& operator=(const IndexerTensorParallel&) = delete;
  IndexerTensorParallel(IndexerTensorParallel&& other) noexcept;
  IndexerTensorParallel& operator=(IndexerTensorParallel&&) = delete;
  static Result<IndexerTensorParallel> Start(const FlashConfig& config, std::uint32_t layer,
      const IndexerPipelineLaunch& launch,
      std::uint32_t rank, std::uintptr_t communicator,
      const EngramCompletionResources& completion, Clock::time_point deadline);
  Result<IndexerPipelineState> Advance();
  IndexerPipelineState state() const noexcept { return state_; }
 private:
  IndexerTensorParallel() = default;
  IndexerPipelineLaunch launch_{};
  std::uint32_t rank_ = 0;
  std::uintptr_t communicator_ = 0;
  Clock::time_point deadline_{};
  EngramCompletionResources resources_{};
  std::optional<EngramReduction> reduction_;
  std::optional<EngramCompletion> completion_;
  IndexerPipelineState state_ = IndexerPipelineState::kFailed;
};
}  // namespace pih::deepseek_v41
