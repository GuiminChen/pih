#pragma once
#include "expert_workspace.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih::deepseek_v41 {
enum class ExpertWorkspaceState { kEmpty, kReady, kReserved, kInUse, kWaitingFence, kReleased, kQuarantined };
// Explicit-release allocation ledger. The capability activations, context,
// stream and exclusive retirement event are borrowed and outlive this owner.
// Not thread-safe. Destruction never frees potentially in-flight storage;
// caller must Release successfully or transfer the ledger to fault retirement.
class ExpertWorkspaceOwner final {
 public:
  ExpertWorkspaceOwner(const pih_nvidia_cuda_memory_api_v1& memory,
      const pih_nvidia_cuda_async_api_v1& async, std::int32_t device,
      std::uintptr_t context, std::uintptr_t stream, std::uintptr_t event);
  ExpertWorkspaceOwner(const ExpertWorkspaceOwner&) = delete;
  ExpertWorkspaceOwner& operator=(const ExpertWorkspaceOwner&) = delete;
  ExpertWorkspaceOwner(ExpertWorkspaceOwner&&) = delete;
  ExpertWorkspaceOwner& operator=(ExpertWorkspaceOwner&&) = delete;
  Status Allocate(std::uint32_t tokens);
  Status ValidateBinding(std::uint32_t tokens, std::uintptr_t stream,
      std::uintptr_t computation_event, std::uint64_t reservation = 0) const;
  Result<std::uint64_t> Reserve();
  Result<EngramDeviceRegion> BeginUse(std::uint64_t reservation = 0);
  // Only after ALL users' final commands have been enqueued on the bound stream
  // and NCCL has no pending enqueue. No new work may use the region afterwards.
  Status RecordRetirementFence(std::uint64_t reservation = 0);
  Result<bool> PollRetirement();
  Status Release();
  ExpertWorkspaceState state() const noexcept { return state_; }
  const pih_cuda_allocation_v1& allocation_record() const noexcept { return allocation_; }
  // For fault-retirement reconciliation, not an invitation to share a live lease.
  std::uint64_t reservation_record() const noexcept { return reservation_; }
 private:
  const pih_nvidia_cuda_memory_api_v1* memory_;
  const pih_nvidia_cuda_async_api_v1* async_;
  std::int32_t device_;
  std::uintptr_t context_, stream_, event_;
  std::uint32_t tokens_ = 0;
  std::uint64_t next_reservation_ = 0, reservation_ = 0;
  pih_cuda_allocation_v1 allocation_{sizeof(pih_cuda_allocation_v1), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1};
  ExpertWorkspaceState state_ = ExpertWorkspaceState::kEmpty;
};
}  // namespace pih::deepseek_v41
