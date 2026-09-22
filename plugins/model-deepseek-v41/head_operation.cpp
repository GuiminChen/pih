#include "head_operation.h"
#include <array>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
HeadOperation::HeadOperation(HeadOperation&& other) noexcept
    : sequence_(other.sequence_), logits_(other.logits_), error_(other.error_), stream_(other.stream_), step_end_(other.step_end_),
      resources_(other.resources_), deadline_(other.deadline_), gather_(std::move(other.gather_)),
      completion_(std::move(other.completion_)), state_(other.state_) {
  other.sequence_ = nullptr; other.state_ = HeadOperationState::kFailed;
}
HeadOperation::~HeadOperation() {
  if (sequence_ && state_ != HeadOperationState::kComplete) sequence_->Fail();
}
Result<HeadOperation> HeadOperation::Start(BlockSequence& sequence, ModelHeadLaunch x, const BackboneWeightUpload& weights,
    EngramDeviceRegion logits, std::uintptr_t communicator,
    const EngramCompletionResources& resources, Clock::time_point deadline) {
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Model head deadline expired before admission");
  const auto step = sequence.StepOutput(); if (!step.ok()) return step.status();
  auto arena = weights.Arena(); if (!arena.ok()) return arena.status();
  if (!Same(sequence.weight_arena_, *arena) || sequence.config_.config_sha256() != weights.catalog().config_sha256())
    return Status::InvalidArgument("Head weights differ from sequence config or arena");
  auto norm = weights.Find("norm.weight"); if (!norm.ok()) return norm.status();
  x.input.norm.weight = *norm; x.input.norm.weight_storage = EngramStorage::kBF16;
  auto& c = x.input.collapse; auto& n = x.input.norm; auto& p = x.projection;
  if (sequence.head_end_ == sequence.start_ || c.tokens != step->tokens || n.rows != step->tokens ||
      n.width != 5120 || p.world_size != sequence.world_ || p.rank != sequence.rank_ ||
      p.stream != sequence.stream_ || !Same(p.error_flag, sequence.error_) || communicator != sequence.communicator_)
    return Status::InvalidArgument("Model head step, sequence rank/stream or one-shot publication mismatch");
  c.residual = step->residual; c.pre = step->pre;
  // Validate normalization extents before deriving its last-row view.
  const auto input = ValidateMhcInput(x.input); if (!input.ok()) return input;
  p.input = {n.output.address + (n.rows - 1ULL) * 5120 * 2, 5120ULL * 2};
  auto bound = BindUploadedHead(weights, x); if (!bound.ok()) return bound.status();
  x = *bound;
  if (logits.bytes != 129280ULL * 4)
    return Status::InvalidArgument("Model head requires a full 129280-entry FP32 vocabulary output");
  const Fp32GatherLaunch gather{p.output, logits, p.stream, p.world_size, p.rank};
  const auto transport = ValidateFp32Gather(gather, communicator); if (!transport.ok()) return transport;
  const auto completion = ValidateEngramCompletionResources(resources); if (!completion.ok()) return completion;
  const std::array reads{c.residual, c.pre, n.weight, p.weight};
  const std::array writes{c.output, n.output, p.output, p.error_flag};
  for (const auto read : reads) if (Overlap(logits, read))
    return Status::InvalidArgument("Gathered logits overwrite head input or weight");
  for (const auto write : writes) if (Overlap(logits, write))
    return Status::InvalidArgument("Gathered logits overlap local head storage");
  for (const auto retained : sequence.Retained(true)) {
    if (Overlap(logits, retained)) return Status::InvalidArgument("Gathered logits overwrite a sequence cache");
    for (const auto write : writes) if (Overlap(write, retained))
      return Status::InvalidArgument("Model head overwrites a sequence cache");
  }
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Model head deadline expired during admission");
  HeadOperation operation;
  operation.sequence_ = &sequence; operation.logits_ = logits; operation.error_ = p.error_flag;
  operation.step_end_ = sequence.start_;
  operation.stream_ = p.stream; operation.resources_ = resources; operation.deadline_ = deadline;
  sequence.active_ = true;
  const auto launched = LaunchModelHead(x); if (!launched.ok()) return launched;
  auto submitted = EngramReduction::SubmitGatherFp32(gather, communicator); if (!submitted.ok()) return submitted.status();
  operation.gather_.emplace(std::move(*submitted)); operation.state_ = HeadOperationState::kWaitingGather;
  return operation;
}
Result<HeadOperationState> HeadOperation::Advance() {
  auto result = AdvanceImpl();
  if (!result.ok() && sequence_) sequence_->Fail();
  return result;
}
Result<HeadOperationState> HeadOperation::AdvanceImpl() {
  if (state_ == HeadOperationState::kFailed || !sequence_)
    return Status::FailedPrecondition("Model head failed or moved from");
  if (state_ == HeadOperationState::kComplete) return state_;
  const auto previous = state_; state_ = HeadOperationState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Model head deadline expired; retire generation");
  const auto gathered = gather_->PollEnqueued(); if (!gathered.ok()) return gathered.status();
  if (*gathered == EngramReductionState::kPending) {
    if (previous != HeadOperationState::kWaitingGather)
      return Status::FailedPrecondition("Model head communicator reused before completion");
    state_ = previous; return state_;
  }
  if (previous == HeadOperationState::kWaitingGather) {
    auto completion = EngramCompletion::RecordFlag(error_, stream_, resources_); if (!completion.ok()) return completion.status();
    completion_.emplace(std::move(*completion)); state_ = HeadOperationState::kWaitingCompletion; return state_;
  }
  const auto ready = completion_->Poll(); if (!ready.ok()) return ready.status();
  if (!*ready) { state_ = previous; return state_; }
  const auto final = gather_->PollEnqueued(); if (!final.ok()) return final.status();
  if (*final != EngramReductionState::kEnqueued)
    return Status::FailedPrecondition("Model head communicator changed at completion");
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Model head deadline expired at completion");
  sequence_->active_ = false; sequence_->head_end_ = sequence_->start_;
  state_ = HeadOperationState::kComplete; return state_;
}
Result<EngramDeviceRegion> HeadOperation::Logits() const {
  if (state_ != HeadOperationState::kComplete || !sequence_ || sequence_->failed_ || sequence_->active_ ||
      sequence_->input_ready_ || sequence_->layer_ || sequence_->start_ != step_end_ || sequence_->head_end_ != step_end_)
    return Status::FailedPrecondition("Model head logits are incomplete or superseded");
  return logits_;
}
}  // namespace pih::deepseek_v41
