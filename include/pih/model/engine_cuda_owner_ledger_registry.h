#pragma once

#include <map>
#include <mutex>

#include "pih/model/engine_cuda_owner_ledger_operations.h"

namespace pih {

class EngineCudaOwnerLedgerClock {
 public:
  virtual ~EngineCudaOwnerLedgerClock() = default;
  virtual Result<std::uint64_t> monotonic_ns() = 0;
};

class EngineCudaOwnerLedgerRegistry final
    : public EngineCudaOwnerLedgerBackend {
 public:
  explicit EngineCudaOwnerLedgerRegistry(
      EngineCudaOwnerLedgerClock& clock) noexcept
      : clock_(&clock) {}

  Status register_pinned(std::uint64_t registration_identity,
                         Sha256Digest physical_gpu_identity,
                         std::int32_t numa_node,
                         std::uint64_t registered_bytes);
  Status release_pinned(std::uint64_t registration_identity);
  Status register_allocation(std::uint64_t allocation_identity,
                             Sha256Digest physical_gpu_identity,
                             std::uint64_t allocated_bytes);
  Status release_allocation(std::uint64_t allocation_identity);
  void poison(const char* reason) noexcept;

  Result<EngineCudaOwnerLedgerSnapshot> capture(
      std::span<const std::uint64_t> pinned_identities,
      std::span<const std::uint64_t> allocation_identities) override;

 private:
  EngineCudaOwnerLedgerClock* clock_ = nullptr;
  std::mutex mutex_;
  std::map<std::uint64_t, EnginePinnedMemoryObservation> pinned_;
  std::map<std::uint64_t, EngineGpuAllocationObservation> allocations_;
  std::uint64_t next_sample_identity_ = 1;
  bool poisoned_ = false;
};

}  // namespace pih
