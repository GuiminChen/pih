#include "pih/model/engine_artifact_descriptor_inventory.h"

#include <set>

namespace pih {
namespace {

bool zero_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) {
    if (value != std::byte{0}) return false;
  }
  return true;
}

}  // namespace

Result<EngineArtifactDescriptorInventory>
EngineArtifactDescriptorInventory::Create(
    std::span<const EngineArtifactDescriptorBinding> bindings,
    EngineArtifactDescriptorOperations& operations) {
  if (bindings.empty()) {
    return Status::InvalidArgument(
        "engine artifact descriptor manifest is empty");
  }
  std::set<std::uint64_t> descriptor_identities;
  std::set<std::string> owner_resource_identities;
  for (const auto& binding : bindings) {
    if (binding.descriptor_identity == 0 ||
        zero_digest(binding.owner_identity) ||
        zero_digest(binding.resource_identity) ||
        binding.expected_device_identity == 0 ||
        binding.expected_inode_identity == 0 ||
        binding.expected_size_bytes == 0 ||
        static_cast<std::uint32_t>(binding.state) >= 3 ||
        !descriptor_identities.insert(binding.descriptor_identity).second ||
        !owner_resource_identities
             .insert(binding.owner_identity.hex() +
                     binding.resource_identity.hex())
             .second) {
      return Status::InvalidArgument(
          "engine artifact descriptor manifest is invalid");
    }
  }
  return EngineArtifactDescriptorInventory(
      std::vector<EngineArtifactDescriptorBinding>(bindings.begin(),
                                                   bindings.end()),
      operations);
}

Result<std::vector<EngineOwnedResourceManifestRecord>>
EngineArtifactDescriptorInventory::capture(EngineOwnedResourceKind kind) {
  if (kind != EngineOwnedResourceKind::kArtifact) {
    return Status::InvalidArgument(
        "engine artifact descriptor inventory kind is invalid");
  }
  std::vector<EngineOwnedResourceManifestRecord> records;
  records.reserve(bindings_.size());
  for (const auto& binding : bindings_) {
    auto observation = operations_->observe(binding.descriptor_identity);
    if (!observation.ok()) return observation.status();
    if (observation->descriptor_identity != binding.descriptor_identity) {
      return Status::FailedPrecondition(
          "engine artifact descriptor identity drifted");
    }
    if (!observation->open) continue;
    if (!observation->regular_file ||
        observation->device_identity != binding.expected_device_identity ||
        observation->inode_identity != binding.expected_inode_identity ||
        observation->size_bytes != binding.expected_size_bytes) {
      return Status::FailedPrecondition(
          "engine artifact descriptor stat identity drifted");
    }
    records.push_back({EngineOwnedResourceKind::kArtifact,
                       binding.owner_identity, binding.resource_identity,
                       observation->size_bytes, 1, binding.state});
  }
  return records;
}

}  // namespace pih
