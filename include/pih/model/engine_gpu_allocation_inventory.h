#pragma once

#include <span>

#include "pih/model/engine_owned_resource_manifest_census_source.h"

namespace pih {

struct EngineGpuAllocationBinding final {
  std::uint64_t allocation_identity = 0;
  Sha256Digest owner_identity{};
  Sha256Digest resource_identity{};
  Sha256Digest physical_gpu_identity{};
  std::uint64_t allocated_bytes = 0;
  EngineOwnedResourceLifecycleState state =
      EngineOwnedResourceLifecycleState::kActive;
};

struct EngineGpuAllocationObservation final {
  std::uint64_t allocation_identity = 0;
  bool allocated = false;
  bool owner_counter_visible = false;
  Sha256Digest physical_gpu_identity{};
  std::uint64_t allocated_bytes = 0;
};

class EngineGpuAllocationOperations {
 public:
  virtual ~EngineGpuAllocationOperations() = default;
  virtual Result<std::vector<EngineGpuAllocationObservation>> capture_allocations(
      std::span<const std::uint64_t> allocation_identities) = 0;
};

class EngineGpuAllocationInventory final
    : public EngineOwnedResourceManifestInventory {
 public:
  static Result<EngineGpuAllocationInventory> Create(
      std::span<const EngineGpuAllocationBinding> bindings,
      EngineGpuAllocationOperations& operations);
  Result<std::vector<EngineOwnedResourceManifestRecord>> capture(
      EngineOwnedResourceKind kind) override;

 private:
  EngineGpuAllocationInventory(
      std::vector<EngineGpuAllocationBinding> bindings,
      EngineGpuAllocationOperations& operations) noexcept
      : bindings_(std::move(bindings)), operations_(&operations) {}
  std::vector<EngineGpuAllocationBinding> bindings_;
  EngineGpuAllocationOperations* operations_ = nullptr;
};

}  // namespace pih
