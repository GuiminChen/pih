#include "attention_pipeline.h"
#include <utility>

namespace pih::deepseek_v41 {
AttentionTensorParallel::AttentionTensorParallel(AttentionTensorParallel&& other) noexcept
    : launch_(other.launch_), rank_(other.rank_), communicator_(other.communicator_), resources_(other.resources_),
      deadline_(other.deadline_), reduction_(std::move(other.reduction_)), completion_(std::move(other.completion_)), state_(other.state_) {
  other.communicator_ = 0; other.state_ = AttentionPipelineState::kFailed;
}
Result<AttentionTensorParallel> AttentionTensorParallel::Start(const AttentionResidualLaunch& launch,
    std::uint32_t rank, std::uintptr_t communicator, const EngramCompletionResources& resources, Clock::time_point deadline) {
  const auto validation = ValidateAttentionResidual(launch); if (!validation.ok()) return validation;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Attention deadline expired before submission");
  const auto transport = ValidateAttentionReduction(launch.attention.output, rank, communicator); if (!transport.ok()) return transport;
  const auto completion = ValidateEngramCompletionResources(resources); if (!completion.ok()) return completion;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Attention deadline expired during preflight");
  const auto attention = LaunchAssembledAttentionOutput(launch.attention); if (!attention.ok()) return attention;
  auto reduction = ReduceAttentionOutput(launch.attention.output, rank, communicator); if (!reduction.ok()) return reduction.status();
  AttentionTensorParallel operation;
  operation.launch_ = launch; operation.rank_ = rank; operation.communicator_ = communicator;
  operation.resources_ = resources; operation.deadline_ = deadline;
  operation.reduction_.emplace(std::move(*reduction)); operation.state_ = AttentionPipelineState::kWaitingReduction;
  return operation;
}
Result<AttentionPipelineState> AttentionTensorParallel::Advance() {
  if (state_ == AttentionPipelineState::kFailed || !communicator_ || !reduction_)
    return Status::FailedPrecondition("Attention operation failed or moved from");
  if (state_ == AttentionPipelineState::kComplete) return state_;
  const auto previous = state_;
  state_ = AttentionPipelineState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Attention deadline expired; owner must abort generation");
  const auto reduced = reduction_->PollEnqueued(); if (!reduced.ok()) return reduced.status();
  if (*reduced == EngramReductionState::kPending) {
    if (previous != AttentionPipelineState::kWaitingReduction)
      return Status::FailedPrecondition("Attention communicator reused before completion");
    state_ = previous; return state_;
  }
  if (previous == AttentionPipelineState::kWaitingCompletion) {
    const auto complete = completion_->Poll(); if (!complete.ok()) return complete.status();
    if (!*complete) { state_ = previous; return state_; }
    const auto transport = reduction_->PollEnqueued(); if (!transport.ok()) return transport.status();
    if (*transport != EngramReductionState::kEnqueued) return Status::FailedPrecondition("Attention communicator changed at completion");
    if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Attention deadline expired at completion");
    state_ = AttentionPipelineState::kComplete; return state_;
  }
  // Polling may occur on another host thread: re-admit its CUDA device/rank.
  const auto transport = ValidateAttentionReduction(launch_.attention.output, rank_, communicator_); if (!transport.ok()) return transport;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Attention deadline expired before residual submission");
  const auto rounded = RoundAttentionOutput(launch_.attention.output); if (!rounded.ok()) return rounded;
  const auto residual = LaunchMhcPost(launch_.residual); if (!residual.ok()) return residual;
  auto completion = EngramCompletion::RecordFlag(launch_.residual.error_flag, launch_.residual.stream, resources_);
  if (!completion.ok()) return completion.status();
  completion_.emplace(std::move(*completion)); state_ = AttentionPipelineState::kWaitingCompletion;
  return state_;
}
}  // namespace pih::deepseek_v41
