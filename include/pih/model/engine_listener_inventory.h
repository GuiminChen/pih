#pragma once

#include <span>

#include "pih/model/engine_owned_resource_manifest_census_source.h"

namespace pih {

struct EngineListenerBinding final {
  std::uint64_t listener_identity = 0;
  Sha256Digest owner_identity{};
  Sha256Digest resource_identity{};
  std::uint64_t expected_device_identity = 0;
  std::uint64_t expected_inode_identity = 0;
  bool expected_kernel_listening = false;
};

struct EngineListenerObservation final {
  std::uint64_t listener_identity = 0;
  bool open = false;
  bool socket = false;
  bool kernel_listening = false;
  bool accept_authority = false;
  std::uint64_t device_identity = 0;
  std::uint64_t inode_identity = 0;
};

class EngineListenerOperations {
 public:
  virtual ~EngineListenerOperations() = default;
  virtual Result<EngineListenerObservation> observe(
      std::uint64_t listener_identity) = 0;
};

class EngineListenerInventory final
    : public EngineOwnedResourceManifestInventory {
 public:
  static Result<EngineListenerInventory> Create(
      std::span<const EngineListenerBinding> bindings,
      EngineListenerOperations& operations);

  Result<std::vector<EngineOwnedResourceManifestRecord>> capture(
      EngineOwnedResourceKind kind) override;

 private:
  EngineListenerInventory(std::vector<EngineListenerBinding> bindings,
                          EngineListenerOperations& operations) noexcept
      : bindings_(std::move(bindings)), operations_(&operations) {}

  std::vector<EngineListenerBinding> bindings_;
  EngineListenerOperations* operations_ = nullptr;
};

}  // namespace pih
