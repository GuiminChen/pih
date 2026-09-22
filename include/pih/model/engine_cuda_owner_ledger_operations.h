#pragma once

#include <mutex>
#include <memory>
#include <span>

#include "pih/model/engine_gpu_allocation_inventory.h"
#include "pih/model/engine_pinned_memory_inventory.h"

namespace pih {

struct EngineCudaOwnerLedgerSnapshot final {
  std::uint64_t sample_identity = 0;
  std::uint64_t sample_started_ns = 0;
  std::uint64_t sample_completed_ns = 0;
  std::vector<EnginePinnedMemoryObservation> pinned;
  std::vector<EngineGpuAllocationObservation> allocations;
};

class EngineCudaOwnerLedgerBackend {
 public:
  virtual ~EngineCudaOwnerLedgerBackend() = default;
  virtual Result<EngineCudaOwnerLedgerSnapshot> capture(
      std::span<const std::uint64_t> pinned_identities,
      std::span<const std::uint64_t> allocation_identities) = 0;
};

class EngineCudaOwnerLedgerOperations final
    : public EnginePinnedMemoryOperations,
      public EngineGpuAllocationOperations {
 public:
  static Result<std::unique_ptr<EngineCudaOwnerLedgerOperations>> Create(
      std::span<const std::uint64_t> pinned_identities,
      std::span<const std::uint64_t> allocation_identities,
      EngineCudaOwnerLedgerBackend& backend);

  Result<std::vector<EnginePinnedMemoryObservation>> capture_pinned(
      std::span<const std::uint64_t> registration_identities) override;
  Result<std::vector<EngineGpuAllocationObservation>> capture_allocations(
      std::span<const std::uint64_t> allocation_identities) override;

 private:
  EngineCudaOwnerLedgerOperations(
      std::vector<std::uint64_t> pinned_identities,
      std::vector<std::uint64_t> allocation_identities,
      EngineCudaOwnerLedgerBackend& backend) noexcept
      : pinned_identities_(std::move(pinned_identities)),
        allocation_identities_(std::move(allocation_identities)),
        backend_(&backend) {}

  Result<EngineCudaOwnerLedgerSnapshot> sample_locked();
  Status poison_locked(const char* message);

  std::vector<std::uint64_t> pinned_identities_;
  std::vector<std::uint64_t> allocation_identities_;
  EngineCudaOwnerLedgerBackend* backend_ = nullptr;
  std::mutex mutex_;
  bool poisoned_ = false;
  std::uint64_t last_sample_identity_ = 0;
  std::uint64_t last_sample_completed_ns_ = 0;
};

}  // namespace pih
