#include "pih/model/engine_pinned_memory_inventory.h"

#include <set>

namespace pih {
namespace {
bool zero_pinned_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes)
    if (value != std::byte{0}) return false;
  return true;
}
}  // namespace

Result<EnginePinnedMemoryInventory> EnginePinnedMemoryInventory::Create(
    std::span<const EnginePinnedMemoryBinding> bindings,
    EnginePinnedMemoryOperations& operations) {
  if (bindings.empty())
    return Status::InvalidArgument("pinned memory manifest is empty");
  std::set<std::uint64_t> registrations;
  std::set<std::string> owner_resources;
  for (const auto& binding : bindings) {
    if (binding.registration_identity == 0 ||
        zero_pinned_digest(binding.owner_identity) ||
        zero_pinned_digest(binding.resource_identity) ||
        zero_pinned_digest(binding.physical_gpu_identity) ||
        binding.numa_node < 0 || binding.registered_bytes == 0 ||
        static_cast<std::uint32_t>(binding.state) >= 3 ||
        !registrations.insert(binding.registration_identity).second ||
        !owner_resources
             .insert(binding.owner_identity.hex() +
                     binding.resource_identity.hex())
             .second) {
      return Status::InvalidArgument("pinned memory manifest is invalid");
    }
  }
  return EnginePinnedMemoryInventory(
      std::vector<EnginePinnedMemoryBinding>(bindings.begin(), bindings.end()),
      operations);
}

Result<std::vector<EngineOwnedResourceManifestRecord>>
EnginePinnedMemoryInventory::capture(EngineOwnedResourceKind kind) {
  if (kind != EngineOwnedResourceKind::kPinnedMemory)
    return Status::InvalidArgument("pinned memory inventory kind is invalid");
  std::vector<std::uint64_t> identities;
  identities.reserve(bindings_.size());
  for (const auto& binding : bindings_)
    identities.push_back(binding.registration_identity);
  auto observations = operations_->capture_pinned(identities);
  if (!observations.ok()) return observations.status();
  if (observations->size() != bindings_.size())
    return Status::FailedPrecondition(
        "pinned memory snapshot cardinality drifted");
  std::vector<EngineOwnedResourceManifestRecord> records;
  records.reserve(bindings_.size());
  for (std::size_t index = 0; index < bindings_.size(); ++index) {
    const auto& binding = bindings_[index];
    const auto& observation = observations->at(index);
    if (observation.registration_identity != binding.registration_identity)
      return Status::FailedPrecondition(
          "pinned memory registration identity drifted");
    if (!observation.owner_counter_visible)
      return Status::Unavailable("pinned memory owner counter is unavailable");
    if (!observation.registered) {
      if (observation.registered_bytes != 0)
        return Status::FailedPrecondition(
            "unregistered pinned memory retained bytes");
      continue;
    }
    if (observation.physical_gpu_identity !=
            binding.physical_gpu_identity ||
        observation.numa_node != binding.numa_node ||
        observation.registered_bytes != binding.registered_bytes) {
      return Status::FailedPrecondition("pinned memory topology drifted");
    }
    records.push_back({EngineOwnedResourceKind::kPinnedMemory,
                       binding.owner_identity, binding.resource_identity,
                       observation.registered_bytes, 1, binding.state});
  }
  return records;
}
}  // namespace pih
