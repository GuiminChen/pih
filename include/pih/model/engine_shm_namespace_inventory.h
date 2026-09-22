#pragma once

#include <span>

#include "pih/model/engine_owned_resource_manifest_census_source.h"

namespace pih {

struct EngineShmObjectBinding final {
  std::uint64_t object_identity = 0;
  Sha256Digest owner_identity{};
  Sha256Digest resource_identity{};
  std::uint64_t expected_device_identity = 0;
  std::uint64_t expected_inode_identity = 0;
  std::uint64_t expected_size_bytes = 0;
  EngineOwnedResourceLifecycleState state =
      EngineOwnedResourceLifecycleState::kActive;
};

struct EngineShmObjectObservation final {
  std::uint64_t object_identity = 0;
  bool present = false;
  bool regular_file = false;
  std::uint64_t device_identity = 0;
  std::uint64_t inode_identity = 0;
  std::uint64_t size_bytes = 0;
};

class EngineShmNamespaceOperations {
 public:
  virtual ~EngineShmNamespaceOperations() = default;
  virtual Result<EngineShmObjectObservation> observe(
      std::uint64_t object_identity) = 0;
};

class EngineShmNamespaceInventory final
    : public EngineOwnedResourceManifestInventory {
 public:
  static Result<EngineShmNamespaceInventory> Create(
      std::span<const EngineShmObjectBinding> bindings,
      EngineShmNamespaceOperations& operations);

  Result<std::vector<EngineOwnedResourceManifestRecord>> capture(
      EngineOwnedResourceKind kind) override;

 private:
  EngineShmNamespaceInventory(
      std::vector<EngineShmObjectBinding> bindings,
      EngineShmNamespaceOperations& operations) noexcept
      : bindings_(std::move(bindings)), operations_(&operations) {}

  std::vector<EngineShmObjectBinding> bindings_;
  EngineShmNamespaceOperations* operations_ = nullptr;
};

}  // namespace pih
