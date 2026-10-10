#pragma once
#include "weight_memory_owner.h"
#include "packed_inference_memory_owner.h"
#include "worker_handles.h"

namespace pih::deepseek_v41 {
struct PackedWorkerMemoryBudget final {
  std::uint64_t device_bytes = 0, host_bytes = 0, staging_bytes = 1U << 20;
};
struct PackedWorkerMemoryViews final {
  const BackboneWeightUpload& weights;
  PackedInferenceMemoryOwner& inference;
  ExpertWorkspaceOwner& prefill;
  ExpertWorkspaceOwner& decode;
};
enum class PackedWorkerMemoryState { kEmpty, kUploading, kInitializing, kReady, kRetiring, kReleased, kQuarantined };
// Stable-address assembly for one rank and sequence. Dependencies are borrowed.
// Owns the child ledgers, never frees GPU resources in its destructor.
class PackedWorkerMemory final {
 public:
  using Clock = BackboneWeightUpload::Clock;
  PackedWorkerMemory(const BackboneWeightFiles& files, const PackedInferenceMemoryPlan& plan,
      const pih_nvidia_cuda_memory_api_v1& memory, const pih_nvidia_cuda_async_api_v1& async,
      const WorkerHandles& handles, std::int32_t device)
      : files_(files), plan_(plan), memory_(memory), async_(async), handles_(handles), device_(device) {}
  PackedWorkerMemory(const PackedWorkerMemory&) = delete;
  PackedWorkerMemory& operator=(const PackedWorkerMemory&) = delete;
  Status Start(PackedWorkerMemoryBudget budget, Clock::time_point deadline);
  Result<bool> Advance();
  Result<PackedWorkerMemoryViews> Views();
  // After worker loop stops and all NCCL enqueues/consumers have retired.
  Status BeginRetirement(Clock::time_point deadline);
  PackedWorkerMemoryState state() const noexcept { return state_; }
  const WeightMemoryOwner* weight_ledger() const noexcept { return weights_.get(); }
  const PackedInferenceMemoryOwner* inference_ledger() const noexcept { return inference_.get(); }
  const ExpertWorkspaceOwner* prefill_ledger() const noexcept { return prefill_.get(); }
  const ExpertWorkspaceOwner* decode_ledger() const noexcept { return decode_.get(); }
 private:
  Status Fail(Status status);
  Status ValidateRegions() const;
  const BackboneWeightFiles& files_;
  const PackedInferenceMemoryPlan& plan_;
  const pih_nvidia_cuda_memory_api_v1& memory_;
  const pih_nvidia_cuda_async_api_v1& async_;
  const WorkerHandles& handles_;
  std::int32_t device_;
  Clock::time_point deadline_{};
  std::unique_ptr<WeightMemoryOwner> weights_;
  std::unique_ptr<PackedInferenceMemoryOwner> inference_;
  std::unique_ptr<ExpertWorkspaceOwner> prefill_, decode_;
  PackedWorkerMemoryState state_ = PackedWorkerMemoryState::kEmpty;
};
}  // namespace pih::deepseek_v41
