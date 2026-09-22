#include "pih/model/deepseek_physical_device_registry.h"

#include <set>

namespace pih {
namespace {

bool empty(const Sha256Digest& value) noexcept {
  for (const auto byte : value.bytes) if (byte != std::byte{0}) return false;
  return true;
}

}  // namespace

Result<DeepSeekPhysicalDeviceRegistry> DeepSeekPhysicalDeviceRegistry::Create(
    std::span<const DeepSeekPhysicalDeviceRegistryEntry> entries) {
  if (entries.empty() || entries.size() > 4)
    return Status::InvalidArgument("DeepSeek physical device registry size is invalid");
  std::set<std::uint64_t> identities;
  std::set<std::int32_t> ordinals;
  std::set<std::array<std::byte, 32>> commitments;
  for (std::size_t rank = 0; rank < entries.size(); ++rank) {
    const auto& entry = entries[rank];
    if (entry.rank != rank || entry.startup_device_ordinal < 0 ||
        entry.registry_identity == 0 ||
        empty(entry.uuid_commitment) ||
        !ordinals.insert(entry.startup_device_ordinal).second ||
        !identities.insert(entry.registry_identity).second ||
        !commitments.insert(entry.uuid_commitment.bytes).second)
      return Status::InvalidArgument("DeepSeek physical device registry is invalid");
  }
  return DeepSeekPhysicalDeviceRegistry(
      std::vector<DeepSeekPhysicalDeviceRegistryEntry>(entries.begin(), entries.end()));
}

const DeepSeekPhysicalDeviceRegistryEntry* DeepSeekPhysicalDeviceRegistry::rank(
    std::uint32_t rank_value) const noexcept {
  return rank_value < entries_.size() ? &entries_[rank_value] : nullptr;
}

const DeepSeekPhysicalDeviceRegistryEntry*
DeepSeekPhysicalDeviceRegistry::identity(
    std::uint64_t registry_identity) const noexcept {
  for (const auto& entry : entries_)
    if (entry.registry_identity == registry_identity) return &entry;
  return nullptr;
}

}  // namespace pih
