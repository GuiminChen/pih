#include "packed_worker_memory.h"
#include <array>
#include <new>

namespace pih::deepseek_v41 {
Status PackedWorkerMemory::Fail(Status status) {
  state_ = PackedWorkerMemoryState::kQuarantined;
  return status;
}
Status PackedWorkerMemory::Start(PackedWorkerMemoryBudget budget, Clock::time_point deadline) {
  if (state_ != PackedWorkerMemoryState::kEmpty) return Status::FailedPrecondition("Worker memory is single-use");
  if (handles_.state() != WorkerHandlesState::kReady || device_ < 0 || handles_.device_ordinal() != device_ || deadline <= Clock::now() ||
      !budget.staging_bytes || budget.staging_bytes > (1U << 20) || budget.staging_bytes % 256)
    return Status::InvalidArgument("Worker memory handles, deadline or staging invalid");
  if (plan_.boundary().config_sha256() != files_.catalog().config_sha256() ||
      plan_.boundary().world_size() != files_.catalog().world_size() ||
      plan_.boundary().rank() != files_.catalog().rank())
    return Status::FailedPrecondition("Worker weight and inference configuration or rank differ");
  const auto prefill_bytes = ExpertWorkspaceBytes(plan_.boundary().token_capacity());
  if (!prefill_bytes.ok()) return prefill_bytes.status();
  const auto decode_bytes = ExpertWorkspaceBytes(plan_.maximum_sequences()); if (!decode_bytes.ok()) return decode_bytes.status();
  // Subtraction avoids overflow, and rejects over-budget requests before any
  // provider allocation. Both expert allocations coexist, even for prompt=1.
  auto remaining = budget.device_bytes;
  for (const auto bytes : {files_.catalog().device_bytes(), plan_.device_bytes(), *prefill_bytes, *decode_bytes}) {
    if (bytes > remaining) return Status::ResourceExhausted("Aggregate worker device budget exceeded");
    remaining -= bytes;
  }
  if (plan_.host_bytes() > budget.host_bytes || budget.staging_bytes > budget.host_bytes - plan_.host_bytes())
    return Status::ResourceExhausted("Aggregate worker pinned-host budget exceeded");
  state_ = PackedWorkerMemoryState::kQuarantined;
  deadline_ = deadline;
  const auto& h = handles_.ledger();
  try {
    // Construct every owner before initiating allocations; preserve them on
    // any later failure for process-level retirement/reconciliation.
    weights_ = std::make_unique<WeightMemoryOwner>(files_, memory_, async_, device_, h.context, h.stream, h.events[0]);
    inference_ = std::make_unique<PackedInferenceMemoryOwner>(plan_, memory_, async_, device_, h.context, h.stream, h.events[1]);
    prefill_ = std::make_unique<ExpertWorkspaceOwner>(memory_, async_, device_, h.context, h.stream, h.events[2]);
    decode_ = std::make_unique<ExpertWorkspaceOwner>(memory_, async_, device_, h.context, h.stream, h.events[3]);
    const auto status = weights_->Start(files_.catalog().device_bytes(), budget.staging_bytes, deadline);
    if (!status.ok()) return status;
    state_ = PackedWorkerMemoryState::kUploading;
    return Status::Ok();
  } catch (const std::bad_alloc&) { return Fail(Status::ResourceExhausted("Worker memory owners allocation failed")); }
}
Status PackedWorkerMemory::ValidateRegions() const {
  const std::array regions{weights_->device_record(), weights_->host_record(),
      inference_->device_record(), inference_->host_record(),
      prefill_->allocation_record(), decode_->allocation_record()};
  // Each child validates nonzero extents and overflow before reaching here.
  for (std::size_t i = 0; i < regions.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (regions[i].address < regions[j].address + regions[j].bytes &&
          regions[j].address < regions[i].address + regions[i].bytes)
        return Status::FailedPrecondition("Worker allocations overlap across resource owners");
  return Status::Ok();
}
Result<bool> PackedWorkerMemory::Advance() {
  if (state_ == PackedWorkerMemoryState::kReady || state_ == PackedWorkerMemoryState::kReleased) return true;
  if (state_ != PackedWorkerMemoryState::kUploading && state_ != PackedWorkerMemoryState::kInitializing &&
      state_ != PackedWorkerMemoryState::kRetiring)
    return Status::FailedPrecondition("Worker memory has no pending transition");
  if (Clock::now() >= deadline_) return Fail(Status::DeadlineExceeded("Worker memory transition expired"));
  if (handles_.state() != WorkerHandlesState::kReady)
    return Fail(Status::FailedPrecondition("Worker handles changed during memory transition"));
  try {
    if (state_ == PackedWorkerMemoryState::kUploading) {
      const auto uploaded = weights_->Advance(); if (!uploaded.ok()) return Fail(uploaded.status());
      if (!*uploaded) return false;
      auto status = prefill_->Allocate(plan_.boundary().token_capacity()); if (!status.ok()) return Fail(status);
      status = decode_->Allocate(plan_.maximum_sequences()); if (!status.ok()) return Fail(status);
      status = inference_->Allocate(); if (!status.ok()) return Fail(status);
      status = ValidateRegions(); if (!status.ok()) return Fail(status);
      state_ = PackedWorkerMemoryState::kInitializing;
      return false;
    }
    if (state_ == PackedWorkerMemoryState::kInitializing) {
      const auto ready = inference_->Advance(); if (!ready.ok()) return Fail(ready.status());
      if (!*ready) return false;
      if (Clock::now() >= deadline_) return Fail(Status::DeadlineExceeded("Worker initialization expired"));
      state_ = PackedWorkerMemoryState::kReady;
      return true;
    }
    const auto retired = weights_->Advance(); if (!retired.ok()) return Fail(retired.status());
    if (!*retired) return false;
    // Child owners reject release unless every prior operation has retired.
    auto status = inference_->Release(); if (!status.ok()) return Fail(status);
    status = decode_->Release(); if (!status.ok()) return Fail(status);
    status = prefill_->Release(); if (!status.ok()) return Fail(status);
    status = weights_->Release(); if (!status.ok()) return Fail(status);
    state_ = PackedWorkerMemoryState::kReleased;
    return true;
  } catch (const std::bad_alloc&) { return Fail(Status::ResourceExhausted("Worker memory continuation allocation failed")); }
}
Result<PackedWorkerMemoryViews> PackedWorkerMemory::Views() {
  if (state_ != PackedWorkerMemoryState::kReady) return Status::FailedPrecondition("Worker memory is not ready");
  const auto weights = weights_->Upload(); if (!weights.ok()) return weights.status();
  return PackedWorkerMemoryViews{**weights, *inference_, *prefill_, *decode_};
}
Status PackedWorkerMemory::BeginRetirement(Clock::time_point deadline) {
  if (state_ != PackedWorkerMemoryState::kReady || inference_->state() != InferenceMemoryState::kReady ||
      prefill_->state() != ExpertWorkspaceState::kReady || decode_->state() != ExpertWorkspaceState::kReady ||
      deadline <= Clock::now())
    return Status::FailedPrecondition("Worker memory consumers are not retired");
  const auto status = weights_->BeginRetirement(deadline); if (!status.ok()) return Fail(status);
  deadline_ = deadline;
  state_ = PackedWorkerMemoryState::kRetiring;
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
