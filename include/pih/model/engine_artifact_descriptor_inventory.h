#pragma once

#include <span>

#include "pih/model/engine_owned_resource_manifest_census_source.h"

namespace pih {

struct EngineArtifactDescriptorBinding final {
  std::uint64_t descriptor_identity = 0;
  Sha256Digest owner_identity{};
  Sha256Digest resource_identity{};
  std::uint64_t expected_device_identity = 0;
  std::uint64_t expected_inode_identity = 0;
  std::uint64_t expected_size_bytes = 0;
  EngineOwnedResourceLifecycleState state =
      EngineOwnedResourceLifecycleState::kActive;
};

struct EngineArtifactDescriptorObservation final {
  std::uint64_t descriptor_identity = 0;
  bool open = false;
  bool regular_file = false;
  std::uint64_t device_identity = 0;
  std::uint64_t inode_identity = 0;
  std::uint64_t size_bytes = 0;
};

class EngineArtifactDescriptorOperations {
 public:
  virtual ~EngineArtifactDescriptorOperations() = default;
  virtual Result<EngineArtifactDescriptorObservation> observe(
      std::uint64_t descriptor_identity) = 0;
};

class EngineArtifactDescriptorInventory final
    : public EngineOwnedResourceManifestInventory {
 public:
  static Result<EngineArtifactDescriptorInventory> Create(
      std::span<const EngineArtifactDescriptorBinding> bindings,
      EngineArtifactDescriptorOperations& operations);

  Result<std::vector<EngineOwnedResourceManifestRecord>> capture(
      EngineOwnedResourceKind kind) override;

 private:
  EngineArtifactDescriptorInventory(
      std::vector<EngineArtifactDescriptorBinding> bindings,
      EngineArtifactDescriptorOperations& operations) noexcept
      : bindings_(std::move(bindings)), operations_(&operations) {}

  std::vector<EngineArtifactDescriptorBinding> bindings_;
  EngineArtifactDescriptorOperations* operations_ = nullptr;
};

}  // namespace pih
