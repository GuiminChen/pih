#include "expert_pipeline.h"
#include <utility>

namespace pih::deepseek_v41 {
ExpertTensorParallel::ExpertTensorParallel(ExpertTensorParallel&& other) noexcept
    : launch_(other.launch_), reduction_launch_(other.reduction_launch_), communicator_(other.communicator_),
      resources_(other.resources_), deadline_(other.deadline_), reduction_(std::move(other.reduction_)),
      completion_(std::move(other.completion_)), state_(other.state_) {
  other.communicator_ = 0; other.state_ = ExpertPipelineState::kFailed;
}
Result<ExpertTensorParallel> ExpertTensorParallel::Start(const ExpertResidualLaunch& launch,
    std::uint32_t world_size, std::uint32_t rank, std::uintptr_t communicator,
    const EngramCompletionResources& resources, Clock::time_point deadline) {
  const auto validation = ValidateExpertResidual(launch); if (!validation.ok()) return validation;
  if (world_size != 2 && world_size != 4 && world_size != 8)
    return Status::InvalidArgument("Expert reduction requires 2, 4 or 8 ranks");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Expert reduction deadline expired");
  const Fp32ReductionLaunch reduction_launch{launch.experts.merge.routed, launch.experts.merge.stream, world_size, rank};
  const auto transport = ValidateFp32Reduction(reduction_launch, communicator); if (!transport.ok()) return transport;
  const auto ready = ValidateEngramCompletionResources(resources); if (!ready.ok()) return ready;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Expert reduction deadline expired during preflight");
  auto reduction = EngramReduction::SubmitFp32(reduction_launch, communicator); if (!reduction.ok()) return reduction.status();
  ExpertTensorParallel operation;
  operation.launch_ = launch; operation.reduction_launch_ = reduction_launch; operation.communicator_ = communicator;
  operation.resources_ = resources; operation.deadline_ = deadline;
  operation.reduction_.emplace(std::move(*reduction)); operation.state_ = ExpertPipelineState::kWaitingReduction;
  return operation;
}
Result<ExpertPipelineState> ExpertTensorParallel::Advance() {
  if (state_ == ExpertPipelineState::kFailed || !communicator_ || !reduction_)
    return Status::FailedPrecondition("Expert reduction failed or moved from");
  if (state_ == ExpertPipelineState::kComplete) return state_;
  const auto previous = state_; state_ = ExpertPipelineState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Expert reduction deadline expired; owner must retire work");
  const auto reduced = reduction_->PollEnqueued(); if (!reduced.ok()) return reduced.status();
  if (*reduced == EngramReductionState::kPending) {
    if (previous != ExpertPipelineState::kWaitingReduction)
      return Status::FailedPrecondition("Expert communicator reused before completion");
    state_ = previous; return state_;
  }
  if (previous == ExpertPipelineState::kWaitingCompletion) {
    const auto complete = completion_->Poll(); if (!complete.ok()) return complete.status();
    if (!*complete) { state_ = previous; return state_; }
    const auto transport = reduction_->PollEnqueued(); if (!transport.ok()) return transport.status();
    if (*transport != EngramReductionState::kEnqueued)
      return Status::FailedPrecondition("Expert communicator changed at completion");
    if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Expert deadline expired at completion");
    state_ = ExpertPipelineState::kComplete; return state_;
  }
  const auto transport = ValidateFp32Reduction(reduction_launch_, communicator_); if (!transport.ok()) return transport;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Expert deadline expired before shared merge");
  const auto merged = LaunchExpertResidual(launch_); if (!merged.ok()) return merged;
  auto completion = EngramCompletion::RecordFlag(launch_.residual.error_flag, launch_.residual.stream, resources_);
  if (!completion.ok()) return completion.status();
  completion_.emplace(std::move(*completion)); state_ = ExpertPipelineState::kWaitingCompletion;
  return state_;
}
}  // namespace pih::deepseek_v41
