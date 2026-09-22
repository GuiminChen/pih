#include "pih/model/engine_listener_inventory.h"

#include <set>

namespace pih {
namespace {

bool zero_listener_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) {
    if (value != std::byte{0}) return false;
  }
  return true;
}

}  // namespace

Result<EngineListenerInventory> EngineListenerInventory::Create(
    std::span<const EngineListenerBinding> bindings,
    EngineListenerOperations& operations) {
  if (bindings.empty()) {
    return Status::InvalidArgument("engine listener manifest is empty");
  }
  std::set<std::uint64_t> listener_identities;
  std::set<std::string> owner_resource_identities;
  for (const auto& binding : bindings) {
    if (binding.listener_identity == 0 ||
        zero_listener_digest(binding.owner_identity) ||
        zero_listener_digest(binding.resource_identity) ||
        binding.expected_device_identity == 0 ||
        binding.expected_inode_identity == 0 ||
        !listener_identities.insert(binding.listener_identity).second ||
        !owner_resource_identities
             .insert(binding.owner_identity.hex() +
                     binding.resource_identity.hex())
             .second) {
      return Status::InvalidArgument("engine listener manifest is invalid");
    }
  }
  return EngineListenerInventory(
      std::vector<EngineListenerBinding>(bindings.begin(), bindings.end()),
      operations);
}

Result<std::vector<EngineOwnedResourceManifestRecord>>
EngineListenerInventory::capture(EngineOwnedResourceKind kind) {
  if (kind != EngineOwnedResourceKind::kListener) {
    return Status::InvalidArgument("engine listener inventory kind is invalid");
  }
  std::vector<EngineOwnedResourceManifestRecord> records;
  records.reserve(bindings_.size());
  for (const auto& binding : bindings_) {
    auto observation = operations_->observe(binding.listener_identity);
    if (!observation.ok()) return observation.status();
    if (observation->listener_identity != binding.listener_identity) {
      return Status::FailedPrecondition("engine listener identity drifted");
    }
    if (!observation->open) continue;
    if (!observation->socket ||
        observation->kernel_listening != binding.expected_kernel_listening ||
        (observation->accept_authority &&
         !observation->kernel_listening) ||
        observation->device_identity != binding.expected_device_identity ||
        observation->inode_identity != binding.expected_inode_identity) {
      return Status::FailedPrecondition("engine listener state drifted");
    }
    records.push_back({
        EngineOwnedResourceKind::kListener, binding.owner_identity,
        binding.resource_identity, 0, 1,
        observation->accept_authority
            ? EngineOwnedResourceLifecycleState::kActive
            : EngineOwnedResourceLifecycleState::kPassive});
  }
  return records;
}

}  // namespace pih
