#include "indexer_pipeline.h"
#include <utility>

namespace pih::deepseek_v41 {
IndexerTensorParallel::IndexerTensorParallel(IndexerTensorParallel&& other) noexcept
    : launch_(other.launch_), rank_(other.rank_), communicator_(other.communicator_), deadline_(other.deadline_),
      resources_(other.resources_), reduction_(std::move(other.reduction_)), completion_(std::move(other.completion_)), state_(other.state_) {
  other.communicator_ = 0; other.state_ = IndexerPipelineState::kFailed;
}
Result<IndexerTensorParallel> IndexerTensorParallel::Start(const FlashConfig& config, std::uint32_t layer,
    const IndexerPipelineLaunch& launch,
    std::uint32_t rank, std::uintptr_t communicator, const EngramCompletionResources& resources, Clock::time_point deadline) {
  const auto validation = ValidateIndexerLayer(config, layer, launch); if (!validation.ok()) return validation;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Indexer deadline expired before submission");
  const auto transport = ValidateIndexerReduction(launch.scoring.score, rank, communicator); if (!transport.ok()) return transport;
  const auto completion = ValidateEngramCompletionResources(resources); if (!completion.ok()) return completion;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Indexer deadline expired during preflight");
  if (launch.key) {
    const auto key = LaunchIndexerKey(*launch.key); if (!key.ok()) return key;
  }
  const auto query = LaunchIndexerQuery(launch.query); if (!query.ok()) return query;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Indexer deadline expired during input preparation");
  const auto score = LaunchIndexerWeightedScore(launch.scoring); if (!score.ok()) return score;
  auto reduction = ReduceIndexerScore(launch.scoring.score, rank, communicator); if (!reduction.ok()) return reduction.status();
  IndexerTensorParallel result;
  result.launch_ = launch; result.rank_ = rank; result.communicator_ = communicator;
  result.resources_ = resources; result.deadline_ = deadline;
  result.reduction_.emplace(std::move(*reduction)); result.state_ = IndexerPipelineState::kWaitingReduction;
  return result;
}
Result<IndexerPipelineState> IndexerTensorParallel::Advance() {
  if (state_ == IndexerPipelineState::kFailed || !communicator_ || !reduction_)
    return Status::FailedPrecondition("Indexer operation failed or moved from");
  if (state_ == IndexerPipelineState::kComplete) return state_;
  const auto previous = state_;
  // Any early return below leaves a terminal failure; successful pending paths
  // explicitly restore their state. No failed submission is ever replayed.
  state_ = IndexerPipelineState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Indexer deadline expired; owner must abort generation");
  const auto reduced = reduction_->PollEnqueued(); if (!reduced.ok()) return reduced.status();
  if (*reduced == EngramReductionState::kPending) {
    if (previous != IndexerPipelineState::kWaitingReduction)
      return Status::FailedPrecondition("Indexer communicator reused before completion");
    state_ = previous; return state_;
  }
  if (previous == IndexerPipelineState::kWaitingCompletion) {
    const auto complete = completion_->Poll(); if (!complete.ok()) return complete.status();
    if (!*complete) { state_ = previous; return state_; }
    const auto final_transport = reduction_->PollEnqueued();
    if (!final_transport.ok()) return final_transport.status();
    if (*final_transport != EngramReductionState::kEnqueued) return Status::FailedPrecondition("Indexer communicator changed at completion");
    if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Indexer deadline expired at completion");
    state_ = IndexerPipelineState::kComplete; return state_;
  }
  const auto transport = ValidateIndexerReduction(launch_.scoring.score, rank_, communicator_); if (!transport.ok()) return transport;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Indexer deadline expired before selection");
  if (launch_.candidates) {
    const auto candidates = LaunchIndexerCandidates(*launch_.candidates); if (!candidates.ok()) return candidates;
  }
  const auto selected = LaunchIndexerSelect(launch_.selection); if (!selected.ok()) return selected;
  auto completion = EngramCompletion::RecordFlag(launch_.selection.error_flag, launch_.selection.stream, resources_);
  if (!completion.ok()) return completion.status();
  completion_.emplace(std::move(*completion)); state_ = IndexerPipelineState::kWaitingCompletion;
  return state_;
}
}  // namespace pih::deepseek_v41
