#include "sampling_operation.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
SamplingOperation::SamplingOperation(SamplingOperation&& other) noexcept
    : sequence_(other.sequence_), host_candidate_(other.host_candidate_), parameters_(other.parameters_), observation_(other.observation_),
      completion_(std::move(other.completion_)),
      deadline_(other.deadline_), failed_(other.failed_), complete_(other.complete_) {
  other.sequence_ = nullptr; other.failed_ = true; other.complete_ = false;
}
SamplingOperation::~SamplingOperation() { if (sequence_ && !complete_) sequence_->Fail(); }
Result<SamplingOperation> SamplingOperation::Start(const HeadOperation& head, SamplingLaunch x,
    const SamplingIdentity& identity, EngramDeviceRegion host_candidate,
    const EngramCompletionResources& resources, Clock::time_point deadline) {
  const auto logits = head.Logits(); if (!logits.ok()) return logits.status();
  auto* sequence = head.sequence_;
  if (!sequence || sequence->failed_ || sequence->active_ || sequence->input_ready_ || sequence->layer_ ||
      sequence->start_ != head.step_end_ || sequence->head_end_ != head.step_end_ ||
      sequence->sample_end_ == head.step_end_ || sequence->rank_ + 1 != sequence->world_)
    return Status::FailedPrecondition("Sampling requires this step's completed head on the last rank, exactly once");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Sampling deadline expired before admission");
  x.logits = *logits; x.error_flag = head.error_; x.stream = head.stream_;
  const auto validation = ValidateSampling(x); if (!validation.ok()) return validation;
  auto retained = sequence->Retained(true);
  retained.push_back(sequence->residual_); retained.push_back(sequence->pre_);
  for (const auto write : {x.scores, x.ids, x.weights, x.reduction, x.stats, x.candidate, x.error_flag})
    for (const auto read : retained) if (Overlap(write, read))
      return Status::InvalidArgument("Sampling scratch overwrites retained sequence state");
  const auto admitted = ValidateEngramCompletionResources(resources); if (!admitted.ok()) return admitted;
  if (!identity.epoch || !identity.plan_seq || !identity.sequence_generation || !identity.sampling_config_id)
    return Status::InvalidArgument("Sampling requires nonzero controller identity fields");
  if (!host_candidate.address || host_candidate.address % alignof(SamplingCandidate) ||
      host_candidate.bytes != sizeof(SamplingCandidate) ||
      host_candidate.bytes > std::numeric_limits<std::uintptr_t>::max() - host_candidate.address ||
      Overlap(host_candidate, resources.host_error_flag))
    return Status::InvalidArgument("Sampling host candidate extent, alignment or alias invalid");
  cudaPointerAttributes attributes{};
  auto error = cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(host_candidate.address));
  if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  if (attributes.type != cudaMemoryTypeHost) return Status::FailedPrecondition("Sampling candidate requires pinned host memory");
  int device = -1;
  error = cudaGetDevice(&device); if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  for (const auto region : {x.logits, x.scores, x.ids, x.weights, x.reduction, x.stats, x.candidate, x.error_flag}) {
    error = cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(region.address));
    if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
      return Status::FailedPrecondition("Sampling buffer belongs to a different device domain");
  }
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Sampling deadline expired during admission");
  SamplingOperation operation; operation.sequence_ = sequence; operation.host_candidate_ = host_candidate; operation.deadline_ = deadline;
  operation.parameters_ = x.parameters; operation.observation_.identity = identity;
  operation.observation_.ordinal = x.parameters.ordinal; operation.observation_.processed_length = head.step_end_;
  sequence->active_ = true;
  const auto launched = LaunchSampling(x); if (!launched.ok()) return launched;
  error = cudaMemcpyAsync(reinterpret_cast<void*>(host_candidate.address), reinterpret_cast<const void*>(x.candidate.address),
      sizeof(SamplingCandidate), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(x.stream));
  if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  auto completion = EngramCompletion::RecordFlag(x.error_flag, x.stream, resources); if (!completion.ok()) return completion.status();
  operation.completion_.emplace(std::move(*completion)); operation.failed_ = false;
  return operation;
}
Result<bool> SamplingOperation::Poll() {
  if (failed_ || !sequence_) return Status::FailedPrecondition("Sampling failed or moved from");
  if (complete_) return true;
  if (Clock::now() >= deadline_) {
    failed_ = true; sequence_->Fail(); return Status::DeadlineExceeded("Sampling deadline expired; retire generation");
  }
  const auto ready = completion_->Poll();
  if (!ready.ok()) { failed_ = true; sequence_->Fail(); return ready.status(); }
  if (!*ready) return false;
  if (Clock::now() >= deadline_) {
    failed_ = true; sequence_->Fail(); return Status::DeadlineExceeded("Sampling deadline expired at completion");
  }
  SamplingCandidate candidate{};
  std::memcpy(&candidate, reinterpret_cast<const void*>(host_candidate_.address), sizeof(candidate));
  const auto validation = ValidateSamplingCandidate(candidate, parameters_);
  if (!validation.ok()) { failed_ = true; sequence_->Fail(); return validation; }
  if (Clock::now() >= deadline_) {
    failed_ = true; sequence_->Fail(); return Status::DeadlineExceeded("Sampling deadline expired during host validation");
  }
  observation_.candidate = candidate;
  sequence_->active_ = false; sequence_->sample_end_ = sequence_->start_; complete_ = true;
  return true;
}
Result<SamplingObservation> SamplingOperation::Candidate() const {
  if (!complete_ || failed_) return Status::FailedPrecondition("Sampling candidate is not complete");
  return observation_;
}
}  // namespace pih::deepseek_v41
