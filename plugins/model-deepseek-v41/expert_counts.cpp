#include "expert_counts.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <limits>

namespace pih::deepseek_v41 {
ExpertCounts::ExpertCounts(ExpertCounts&& other) noexcept
    : completion_(std::move(other.completion_)), host_counts_(other.host_counts_), dispatch_(other.dispatch_), counts_(other.counts_),
      first_(other.first_), local_(other.local_), tokens_(other.tokens_), picks_(other.picks_),
      deadline_(other.deadline_), failed_(other.failed_), complete_(other.complete_) { other.failed_ = true; }
Result<ExpertCounts> ExpertCounts::Start(const ExpertDispatchLaunch& d,
    EngramDeviceRegion host_counts, const EngramCompletionResources& resources, Clock::time_point deadline) {
  const auto layout = ValidateExpertDispatch(d); if (!layout.ok()) return layout;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Expert dispatch deadline expired");
  const auto ready = ValidateEngramCompletionResources(resources); if (!ready.ok()) return ready;
  if (!host_counts.address || host_counts.address % 4 || host_counts.bytes != d.counts.bytes ||
      host_counts.bytes > std::numeric_limits<std::uintptr_t>::max() - host_counts.address ||
      (host_counts.address < resources.host_error_flag.address + 4 &&
       resources.host_error_flag.address < host_counts.address + host_counts.bytes))
    return Status::InvalidArgument("Expert count readback layout or host alias invalid");
  cudaPointerAttributes attributes{};
  auto error = cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(host_counts.address));
  if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  if (attributes.type != cudaMemoryTypeHost) return Status::FailedPrecondition("Expert counts require pinned host storage");
  int device = -1;
  error = cudaGetDevice(&device); if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  for (const auto region : {d.indices, d.counts, d.slots, d.error_flag}) {
    error = cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(region.address));
    if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
      return Status::FailedPrecondition("Expert dispatch buffer belongs to a different device domain");
  }
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Expert dispatch deadline expired during admission");
  const auto launched = LaunchExpertDispatch(d); if (!launched.ok()) return launched;
  error = cudaMemcpyAsync(reinterpret_cast<void*>(host_counts.address), reinterpret_cast<const void*>(d.counts.address),
      host_counts.bytes, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(d.stream));
  if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  auto completion = EngramCompletion::RecordFlag(d.error_flag, d.stream, resources);
  if (!completion.ok()) return completion.status();
  ExpertCounts operation;
  operation.completion_.emplace(std::move(*completion)); operation.host_counts_ = host_counts;
  operation.dispatch_ = d;
  operation.local_ = (d.layer < 40 ? 384U : 128U) / d.world_size;
  operation.first_ = d.rank * operation.local_; operation.tokens_ = d.tokens;
  operation.picks_ = d.layer < 40 ? 6U : 3U;
  operation.deadline_ = deadline; operation.failed_ = false;
  return operation;
}
Result<bool> ExpertCounts::Poll() {
  if (failed_ || !completion_) return Status::FailedPrecondition("Expert counts failed or moved from");
  if (complete_) return true;
  failed_ = true;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Expert count observation deadline expired");
  const auto ready = completion_->Poll(); if (!ready.ok()) return ready.status();
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Expert count observation deadline expired at completion");
  if (!*ready) { failed_ = false; return false; }
  std::memcpy(counts_.data(), reinterpret_cast<const void*>(host_counts_.address), local_ * 4ULL);
  std::uint32_t sum = 0;
  for (std::uint32_t i = 0; i < local_; ++i) {
    if (counts_[i] > tokens_) return Status::FailedPrecondition("Expert count exceeds token capacity");
    sum += counts_[i];
  }
  if (sum > tokens_ * picks_) return Status::FailedPrecondition("Expert count total exceeds routing capacity");
  failed_ = false; complete_ = true; return true;
}
Result<std::uint32_t> ExpertCounts::Rows(std::uint32_t expert) const {
  if (failed_ || !complete_) return Status::FailedPrecondition("Expert counts not admitted");
  if (expert < first_ || expert >= first_ + local_) return Status::InvalidArgument("Expert outside observed rank range");
  return counts_[expert - first_];
}
Status ExpertCounts::ValidatePlan(const ExpertDispatchLaunch& d) const {
  if (failed_ || !complete_) return Status::FailedPrecondition("Expert counts not admitted");
  const auto same = [](EngramDeviceRegion a, EngramDeviceRegion b) {
    return a.address == b.address && a.bytes == b.bytes;
  };
  const auto& p = dispatch_;
  if (d.tokens != p.tokens || d.layer != p.layer || d.world_size != p.world_size || d.rank != p.rank ||
      d.stream != p.stream || !same(d.indices, p.indices) || !same(d.counts, p.counts) ||
      !same(d.slots, p.slots) || !same(d.error_flag, p.error_flag))
    return Status::InvalidArgument("Expert dispatch differs from observed count plan");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
