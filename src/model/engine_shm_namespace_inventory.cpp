#include "pih/model/engine_shm_namespace_inventory.h"

#include <set>

namespace pih {
namespace {

bool zero_shm_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) {
    if (value != std::byte{0}) return false;
  }
  return true;
}

}  // namespace

Result<EngineShmNamespaceInventory> EngineShmNamespaceInventory::Create(
    std::span<const EngineShmObjectBinding> bindings,
    EngineShmNamespaceOperations& operations) {
  if (bindings.empty()) {
    return Status::InvalidArgument("engine SHM object manifest is empty");
  }
  std::set<std::uint64_t> object_identities;
  std::set<std::string> owner_resource_identities;
  for (const auto& binding : bindings) {
    if (binding.object_identity == 0 ||
        zero_shm_digest(binding.owner_identity) ||
        zero_shm_digest(binding.resource_identity) ||
        binding.expected_device_identity == 0 ||
        binding.expected_inode_identity == 0 ||
        binding.expected_size_bytes == 0 ||
        static_cast<std::uint32_t>(binding.state) >= 3 ||
        !object_identities.insert(binding.object_identity).second ||
        !owner_resource_identities
             .insert(binding.owner_identity.hex() +
                     binding.resource_identity.hex())
             .second) {
      return Status::InvalidArgument("engine SHM object manifest is invalid");
    }
  }
  return EngineShmNamespaceInventory(
      std::vector<EngineShmObjectBinding>(bindings.begin(), bindings.end()),
      operations);
}

Result<std::vector<EngineOwnedResourceManifestRecord>>
EngineShmNamespaceInventory::capture(EngineOwnedResourceKind kind) {
  if (kind != EngineOwnedResourceKind::kShm) {
    return Status::InvalidArgument("engine SHM inventory kind is invalid");
  }
  std::vector<EngineOwnedResourceManifestRecord> records;
  records.reserve(bindings_.size());
  for (const auto& binding : bindings_) {
    auto observation = operations_->observe(binding.object_identity);
    if (!observation.ok()) return observation.status();
    if (observation->object_identity != binding.object_identity) {
      return Status::FailedPrecondition("engine SHM object identity drifted");
    }
    if (!observation->present) continue;
    if (!observation->regular_file ||
        observation->device_identity != binding.expected_device_identity ||
        observation->inode_identity != binding.expected_inode_identity ||
        observation->size_bytes != binding.expected_size_bytes) {
      return Status::FailedPrecondition("engine SHM object metadata drifted");
    }
    records.push_back({EngineOwnedResourceKind::kShm, binding.owner_identity,
                       binding.resource_identity, observation->size_bytes, 1,
                       binding.state});
  }
  return records;
}

}  // namespace pih
