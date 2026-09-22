#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <span>

#include "pih/core/allocator.h"
#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/model/engine_cuda_owner_ledger_registry.h"

namespace pih {

struct EngineTrackedGpuAllocationPlan final {
  std::uint64_t allocation_identity = 0;
  Sha256Digest physical_gpu_identity{};
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
};

class EngineTrackedGpuAllocator final : public Allocator {
 public:
  static Result<std::unique_ptr<EngineTrackedGpuAllocator>> Create(
      std::span<const EngineTrackedGpuAllocationPlan> plans,
      Allocator& allocator, EngineCudaOwnerLedgerRegistry& registry);
  ~EngineTrackedGpuAllocator() override;

  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override;
  void deallocate(Allocation allocation) noexcept override;

 private:
  struct Active final {
    Allocation allocation;
    std::uint64_t identity = 0;
  };
  EngineTrackedGpuAllocator(
      std::vector<EngineTrackedGpuAllocationPlan> plans,
      Allocator& allocator, EngineCudaOwnerLedgerRegistry& registry) noexcept
      : plans_(std::move(plans)), allocator_(&allocator), registry_(&registry) {}

  std::vector<EngineTrackedGpuAllocationPlan> plans_;
  Allocator* allocator_ = nullptr;
  EngineCudaOwnerLedgerRegistry* registry_ = nullptr;
  std::mutex mutex_;
  std::size_t next_plan_ = 0;
  std::map<std::uint64_t, Active> active_;
};

struct EngineTrackedPinnedAllocationPlan final {
  std::uint64_t registration_identity = 0;
  Sha256Digest physical_gpu_identity{};
  std::int32_t numa_node = -1;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
};

class EngineTrackedPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  static Result<std::unique_ptr<EngineTrackedPinnedAllocator>> Create(
      std::span<const EngineTrackedPinnedAllocationPlan> plans,
      RegisteredPinnedAllocator& allocator,
      EngineCudaOwnerLedgerRegistry& registry);
  ~EngineTrackedPinnedAllocator() override;

  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override;
  void deallocate(Allocation allocation) noexcept override;

 private:
  struct Active final {
    Allocation allocation;
    std::uint64_t identity = 0;
  };
  EngineTrackedPinnedAllocator(
      std::vector<EngineTrackedPinnedAllocationPlan> plans,
      RegisteredPinnedAllocator& allocator,
      EngineCudaOwnerLedgerRegistry& registry) noexcept
      : plans_(std::move(plans)), allocator_(&allocator), registry_(&registry) {}

  std::vector<EngineTrackedPinnedAllocationPlan> plans_;
  RegisteredPinnedAllocator* allocator_ = nullptr;
  EngineCudaOwnerLedgerRegistry* registry_ = nullptr;
  std::mutex mutex_;
  std::size_t next_plan_ = 0;
  std::map<std::uint64_t, Active> active_;
};

}  // namespace pih
