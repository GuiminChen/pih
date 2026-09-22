#pragma once

#include "pih/model/engine_owned_resource_manifest_census_source.h"

namespace pih {

struct EngineNetworkOwnerObservation final {
  Sha256Digest owner_identity{};
  Sha256Digest resource_identity{};
  std::uint64_t backing_bytes = 0;
  std::uint64_t object_count = 0;
  EngineOwnedResourceLifecycleState state =
      EngineOwnedResourceLifecycleState::kActive;
};

struct EngineNetworkNamespaceSnapshot final {
  std::uint64_t namespace_identity = 0;
  bool namespace_handle_present = false;
  bool namespace_destroyed = false;
  std::vector<EngineNetworkOwnerObservation> owners;
};

class EngineNetworkNamespaceOperations {
 public:
  virtual ~EngineNetworkNamespaceOperations() = default;
  virtual Result<EngineNetworkNamespaceSnapshot> capture(
      std::uint64_t namespace_identity) = 0;
};

class EngineNetworkNamespaceInventory final
    : public EngineOwnedResourceManifestInventory {
 public:
  static Result<EngineNetworkNamespaceInventory> Create(
      std::uint64_t namespace_identity,
      EngineNetworkNamespaceOperations& operations);

  Result<std::vector<EngineOwnedResourceManifestRecord>> capture(
      EngineOwnedResourceKind kind) override;

 private:
  EngineNetworkNamespaceInventory(
      std::uint64_t namespace_identity,
      EngineNetworkNamespaceOperations& operations) noexcept
      : namespace_identity_(namespace_identity), operations_(&operations) {}

  std::uint64_t namespace_identity_ = 0;
  EngineNetworkNamespaceOperations* operations_ = nullptr;
};

}  // namespace pih
