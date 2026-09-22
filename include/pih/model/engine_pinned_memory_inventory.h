#pragma once

#include <span>

#include "pih/model/engine_owned_resource_manifest_census_source.h"

namespace pih {

struct EnginePinnedMemoryBinding final {
  std::uint64_t registration_identity = 0;
  Sha256Digest owner_identity{};
  Sha256Digest resource_identity{};
  Sha256Digest physical_gpu_identity{};
  std::int32_t numa_node = -1;
  std::uint64_t registered_bytes = 0;
  EngineOwnedResourceLifecycleState state =
      EngineOwnedResourceLifecycleState::kActive;
};

struct EnginePinnedMemoryObservation final {
  std::uint64_t registration_identity = 0;
  bool registered = false;
  bool owner_counter_visible = false;
  Sha256Digest physical_gpu_identity{};
  std::int32_t numa_node = -1;
  std::uint64_t registered_bytes = 0;
};

class EnginePinnedMemoryOperations {
 public:
  virtual ~EnginePinnedMemoryOperations() = default;
  virtual Result<std::vector<EnginePinnedMemoryObservation>> capture_pinned(
      std::span<const std::uint64_t> registration_identities) = 0;
};

class EnginePinnedMemoryInventory final
    : public EngineOwnedResourceManifestInventory {
 public:
  static Result<EnginePinnedMemoryInventory> Create(
      std::span<const EnginePinnedMemoryBinding> bindings,
      EnginePinnedMemoryOperations& operations);
  Result<std::vector<EngineOwnedResourceManifestRecord>> capture(
      EngineOwnedResourceKind kind) override;

 private:
  EnginePinnedMemoryInventory(
      std::vector<EnginePinnedMemoryBinding> bindings,
      EnginePinnedMemoryOperations& operations) noexcept
      : bindings_(std::move(bindings)), operations_(&operations) {}
  std::vector<EnginePinnedMemoryBinding> bindings_;
  EnginePinnedMemoryOperations* operations_ = nullptr;
};

}  // namespace pih
