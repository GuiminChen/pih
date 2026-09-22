#pragma once

#include <span>

#include "pih/model/engine_artifact_descriptor_inventory.h"

namespace pih {

struct EngineArtifactDescriptorHandleBinding final {
  std::uint64_t descriptor_identity = 0;
  std::int32_t descriptor = -1;
};

struct EngineArtifactDescriptorStat final {
  bool open = false;
  bool regular_file = false;
  std::uint64_t device_identity = 0;
  std::uint64_t inode_identity = 0;
  std::uint64_t size_bytes = 0;
};

class EngineArtifactDescriptorStatBackend {
 public:
  virtual ~EngineArtifactDescriptorStatBackend() = default;
  virtual Result<EngineArtifactDescriptorStat> stat(
      std::int32_t descriptor) = 0;
};

class BoundEngineArtifactDescriptorOperations final
    : public EngineArtifactDescriptorOperations {
 public:
  static Result<BoundEngineArtifactDescriptorOperations> Create(
      std::span<const EngineArtifactDescriptorHandleBinding> bindings,
      EngineArtifactDescriptorStatBackend& backend);

  Result<EngineArtifactDescriptorObservation> observe(
      std::uint64_t descriptor_identity) override;

 private:
  BoundEngineArtifactDescriptorOperations(
      std::vector<EngineArtifactDescriptorHandleBinding> bindings,
      EngineArtifactDescriptorStatBackend& backend) noexcept
      : bindings_(std::move(bindings)), backend_(&backend) {}

  std::vector<EngineArtifactDescriptorHandleBinding> bindings_;
  EngineArtifactDescriptorStatBackend* backend_ = nullptr;
};

}  // namespace pih
