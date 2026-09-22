#include "engram_pipeline.h"
#include <utility>

namespace pih::deepseek_v41 {
EngramTensorParallel::EngramTensorParallel(EngramTensorParallel&& other) noexcept
    : launch_(other.launch_), communicator_(other.communicator_), deadline_(other.deadline_),
      reduction_(std::move(other.reduction_)), completion_resources_(other.completion_resources_),
      completion_(std::move(other.completion_)), state_(other.state_) {
  other.communicator_ = 0;
  other.state_ = EngramPipelineState::kFailed;
}
Result<EngramTensorParallel> EngramTensorParallel::Start(const EngramLaunch& launch,
    std::uintptr_t communicator, const EngramCompletionResources& completion,
    Clock::time_point deadline) {
  const auto chain = ValidateEngramChain(launch);
  if (!chain.ok()) return chain;
  if (Clock::now() >= deadline)
    return Status::DeadlineExceeded("Engram TP deadline expired before submission");
  const auto transport = ValidateEngramReduction(launch.lookup, communicator);
  if (!transport.ok()) return transport;
  const auto completion_status = ValidateEngramCompletionResources(completion);
  if (!completion_status.ok()) return completion_status;
  if (Clock::now() >= deadline)
    return Status::DeadlineExceeded("Engram TP deadline expired during preflight");
  const auto lookup = LaunchEngramLookup(launch.lookup);
  if (!lookup.ok()) return lookup;
  auto reduction = EngramReduction::Submit(launch.lookup, communicator);
  if (!reduction.ok()) return reduction.status();
  EngramTensorParallel operation;
  operation.launch_ = launch;
  operation.communicator_ = communicator;
  operation.deadline_ = deadline;
  operation.completion_resources_ = completion;
  operation.reduction_.emplace(std::move(*reduction));
  operation.state_ = EngramPipelineState::kWaitingReduction;
  return operation;
}
Result<EngramPipelineState> EngramTensorParallel::Advance() {
  if (state_ == EngramPipelineState::kFailed || !communicator_ || !reduction_)
    return Status::FailedPrecondition("Engram TP operation failed or was moved from");
  if (state_ == EngramPipelineState::kComplete) return state_;
  if (Clock::now() >= deadline_) {
    state_ = EngramPipelineState::kFailed;
    return Status::DeadlineExceeded("Engram TP deadline expired; owner must abort generation");
  }
  const auto reduction = reduction_->PollEnqueued();
  if (!reduction.ok()) {
    state_ = EngramPipelineState::kFailed;
    return reduction.status();
  }
  if (*reduction == EngramReductionState::kPending) {
    if (state_ == EngramPipelineState::kWaitingCompletion) {
      state_ = EngramPipelineState::kFailed;
      return Status::FailedPrecondition("Engram communicator reused before operation completion");
    }
    return state_;
  }
  if (state_ == EngramPipelineState::kWaitingCompletion) {
    const auto completed = completion_->Poll();
    if (!completed.ok()) {
      state_ = EngramPipelineState::kFailed;
      return completed.status();
    }
    if (*completed) {
      // Recheck the communicator after the completion event observation, so an
      // async NCCL error observed at retirement cannot authorize this output.
      const auto final_transport = reduction_->PollEnqueued();
      if (!final_transport.ok() || *final_transport != EngramReductionState::kEnqueued) {
        state_ = EngramPipelineState::kFailed;
        return final_transport.ok() ? Status::FailedPrecondition("Engram communicator changed at completion")
                                    : final_transport.status();
      }
      if (Clock::now() >= deadline_) {
        state_ = EngramPipelineState::kFailed;
        return Status::DeadlineExceeded("Engram TP deadline expired at completion observation");
      }
      state_ = EngramPipelineState::kComplete;
    }
    return state_;
  }
  // Re-admit rank/device before launching on a possibly different polling thread.
  const auto transport = ValidateEngramReduction(launch_.lookup, communicator_);
  if (!transport.ok()) {
    state_ = EngramPipelineState::kFailed;
    return transport;
  }
  if (Clock::now() >= deadline_) {
    state_ = EngramPipelineState::kFailed;
    return Status::DeadlineExceeded("Engram TP deadline expired before compute submission");
  }
  const auto projection = LaunchEngramProjection(launch_.projection);
  if (!projection.ok()) {
    state_ = EngramPipelineState::kFailed;
    return projection;
  }
  const auto gate = LaunchEngramGate(launch_.gate);
  if (!gate.ok()) {
    state_ = EngramPipelineState::kFailed;
    return gate;
  }
  auto completion = EngramCompletion::Record(launch_.gate, completion_resources_);
  if (!completion.ok()) {
    state_ = EngramPipelineState::kFailed;
    return completion.status();
  }
  completion_.emplace(std::move(*completion));
  state_ = EngramPipelineState::kWaitingCompletion;
  return state_;
}
}  // namespace pih::deepseek_v41
