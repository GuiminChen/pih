#include "expert_batch.h"
#include <cuda_runtime_api.h>
#include <vector>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateExpertBatch(std::span<const ExpertTokenChainLaunch> experts, const ExpertCounts& counts) {
  if (experts.empty() || experts.size() > 384) return Status::InvalidArgument("Expert batch size invalid");
  const auto& first = experts.front(); const auto& d = first.gather.dispatch;
  const auto layout = ValidateExpertDispatch(d); if (!layout.ok()) return layout;
  const unsigned local = (d.layer < 40 ? 384U : 128U) / d.world_size;
  if (experts.size() != local) return Status::InvalidArgument("Expert batch must include every local expert");
  std::vector<EngramDeviceRegion> reads{d.indices, d.counts, d.slots, first.gather.input, first.gather.route_weights};
  std::vector<EngramDeviceRegion> writes{first.accumulator, d.error_flag};
  for (unsigned i = 0; i < local; ++i) {
    const auto& x = experts[i];
    const auto chain = ValidateExpertTokenChain(x); if (!chain.ok()) return chain;
    const auto plan = counts.ValidatePlan(x.gather.dispatch); if (!plan.ok()) return plan;
    const auto rows = counts.Rows(d.rank * local + i); if (!rows.ok()) return rows.status();
    if (x.gather.expert != d.rank * local + i || x.gather.rows != *rows ||
        !Same(x.gather.input, first.gather.input) || !Same(x.gather.route_weights, first.gather.route_weights) ||
        !Same(x.accumulator, first.accumulator))
      return Status::InvalidArgument("Expert batch order, observed rows or shared input mismatch");
    writes.push_back(x.gather.output); writes.push_back(x.gather.gathered_weights);
    if (x.expert) {
      for (const auto* p : {&x.expert->gate, &x.expert->up, &x.expert->down}) {
        reads.push_back(p->weight); reads.push_back(p->weight_scales);
        writes.push_back(p->quantized); writes.push_back(p->activation_scales); writes.push_back(p->output);
      }
      writes.push_back(x.expert->activation.output);
    }
  }
  // Sequential expert scratch may overlap across chains, but no write may
  // destroy another expert's still-live weights or the common routing inputs.
  for (const auto write : writes) for (const auto read : reads)
    if (Overlap(write, read)) return Status::InvalidArgument("Expert batch overwrites live input or another expert weight");
  return Status::Ok();
}
ExpertBatch::ExpertBatch(ExpertBatch&& other) noexcept
    : completion_(std::move(other.completion_)), deadline_(other.deadline_),
      failed_(other.failed_), complete_(other.complete_) { other.failed_ = true; }
Result<ExpertBatch> ExpertBatch::Start(std::span<const ExpertTokenChainLaunch> experts,
    const ExpertCounts& counts, const EngramCompletionResources& resources, Clock::time_point deadline) {
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Expert batch deadline expired");
  const auto validation = ValidateExpertBatch(experts, counts); if (!validation.ok()) return validation;
  const auto ready = ValidateEngramCompletionResources(resources); if (!ready.ok()) return ready;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Expert batch deadline expired during preflight");
  const auto& first = experts.front(); const auto& d = first.gather.dispatch;
  const auto error = cudaMemsetAsync(reinterpret_cast<void*>(first.accumulator.address), 0,
      first.accumulator.bytes, reinterpret_cast<cudaStream_t>(d.stream));
  if (error != cudaSuccess) return Status::Internal(cudaGetErrorString(error));
  for (const auto& expert : experts) {
    if (Clock::now() >= deadline) return Status::DeadlineExceeded("Expert batch deadline expired during submission");
    const auto submitted = LaunchExpertTokenChain(expert); if (!submitted.ok()) return submitted;
  }
  auto completion = EngramCompletion::RecordFlag(d.error_flag, d.stream, resources);
  if (!completion.ok()) return completion.status();
  ExpertBatch operation;
  operation.completion_.emplace(std::move(*completion)); operation.deadline_ = deadline; operation.failed_ = false;
  return operation;
}
Result<bool> ExpertBatch::Poll() {
  if (failed_ || !completion_) return Status::FailedPrecondition("Expert batch failed or moved from");
  if (complete_) return true;
  failed_ = true;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Expert batch completion deadline expired");
  const auto ready = completion_->Poll(); if (!ready.ok()) return ready.status();
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Expert batch deadline expired at observation");
  failed_ = false; complete_ = *ready; return complete_;
}
}  // namespace pih::deepseek_v41
