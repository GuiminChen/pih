#pragma once
#include "weight_upload.h"
#include "pih/contracts/nvidia_cuda_memory_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih::deepseek_v41 {
enum class WeightMemoryState { kEmpty, kUploading, kReady, kRetiring, kRetired, kReleased, kQuarantined };
// Owns allocations, borrows immutable authenticated files and plugin capabilities.
// No implicit synchronization/free in destruction. Keep this owner and files
// alive until all users of Upload() have stopped and explicit retirement ends.
class WeightMemoryOwner final {
 public:
  using Clock = BackboneWeightUpload::Clock;
  WeightMemoryOwner(const BackboneWeightFiles& files,
      const pih_nvidia_cuda_memory_api_v1& memory, const pih_nvidia_cuda_async_api_v1& async,
      std::int32_t device, std::uintptr_t context, std::uintptr_t stream, std::uintptr_t event)
      : files_(files), memory_(memory), async_(async), device_(device),
        context_(context), stream_(stream), event_(event) {}
  WeightMemoryOwner(const WeightMemoryOwner&) = delete;
  WeightMemoryOwner& operator=(const WeightMemoryOwner&) = delete;
  Status Start(std::uint64_t device_budget, std::uint64_t staging_bytes, Clock::time_point deadline);
  Result<bool> Advance();
  Result<const BackboneWeightUpload*> Upload() const;
  // Stop all consumers first. Every weight use must be on the owned stream,
  // or explicitly joined into it; no pending NCCL enqueue may remain.
  Status BeginRetirement(Clock::time_point deadline);
  Status Release();
  WeightMemoryState state() const noexcept { return state_; }
  const pih_cuda_allocation_v1& device_record() const noexcept { return device_allocation_; }
  const pih_cuda_allocation_v1& host_record() const noexcept { return host_allocation_; }
  bool device_released() const noexcept { return device_released_; }
  bool host_released() const noexcept { return host_released_; }
 private:
  Status Fail(Status status);
  const BackboneWeightFiles& files_;
  const pih_nvidia_cuda_memory_api_v1& memory_;
  const pih_nvidia_cuda_async_api_v1& async_;
  std::int32_t device_;
  std::uintptr_t context_, stream_, event_;
  Clock::time_point deadline_{};
  pih_cuda_allocation_v1 device_allocation_{sizeof(pih_cuda_allocation_v1), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1};
  pih_cuda_allocation_v1 host_allocation_{sizeof(pih_cuda_allocation_v1), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1};
  bool device_released_ = false, host_released_ = false;
  std::unique_ptr<BackboneWeightUpload> upload_;
  WeightMemoryState state_ = WeightMemoryState::kEmpty;
};
}  // namespace pih::deepseek_v41
