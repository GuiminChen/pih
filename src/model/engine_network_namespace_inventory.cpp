#include "pih/model/engine_network_namespace_inventory.h"

#include <set>

namespace pih {
namespace {

bool zero_network_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) {
    if (value != std::byte{0}) return false;
  }
  return true;
}

}  // namespace

Result<EngineNetworkNamespaceInventory>
EngineNetworkNamespaceInventory::Create(
    std::uint64_t namespace_identity,
    EngineNetworkNamespaceOperations& operations) {
  if (namespace_identity == 0) {
    return Status::InvalidArgument(
        "engine network namespace identity is zero");
  }
  return EngineNetworkNamespaceInventory(namespace_identity, operations);
}

Result<std::vector<EngineOwnedResourceManifestRecord>>
EngineNetworkNamespaceInventory::capture(EngineOwnedResourceKind kind) {
  if (kind != EngineOwnedResourceKind::kNetwork) {
    return Status::InvalidArgument(
        "engine network namespace inventory kind is invalid");
  }
  auto snapshot = operations_->capture(namespace_identity_);
  if (!snapshot.ok()) return snapshot.status();
  if (snapshot->namespace_identity != namespace_identity_ ||
      snapshot->namespace_handle_present == snapshot->namespace_destroyed) {
    return Status::FailedPrecondition(
        "engine network namespace state is ambiguous");
  }
  if (snapshot->namespace_destroyed) {
    if (!snapshot->owners.empty()) {
      return Status::FailedPrecondition(
          "destroyed engine network namespace has owners");
    }
    return std::vector<EngineOwnedResourceManifestRecord>{};
  }

  std::set<std::string> owner_resource_states;
  std::vector<EngineOwnedResourceManifestRecord> records;
  records.reserve(snapshot->owners.size());
  for (const auto& owner : snapshot->owners) {
    const auto state = static_cast<std::uint32_t>(owner.state);
    const auto identity = owner.owner_identity.hex() +
                          owner.resource_identity.hex() +
                          std::to_string(state);
    if (zero_network_digest(owner.owner_identity) ||
        zero_network_digest(owner.resource_identity) ||
        owner.object_count == 0 || state >= 2 ||
        !owner_resource_states.insert(identity).second) {
      return Status::FailedPrecondition(
          "engine network namespace owner record is invalid");
    }
    records.push_back({EngineOwnedResourceKind::kNetwork,
                       owner.owner_identity, owner.resource_identity,
                       owner.backing_bytes, owner.object_count, owner.state});
  }
  return records;
}

}  // namespace pih
