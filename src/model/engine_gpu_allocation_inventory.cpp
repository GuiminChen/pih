#include "pih/model/engine_gpu_allocation_inventory.h"

#include <set>

namespace pih {
namespace {
bool zero_allocation_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes)
    if (value != std::byte{0}) return false;
  return true;
}
}  // namespace

Result<EngineGpuAllocationInventory> EngineGpuAllocationInventory::Create(
    std::span<const EngineGpuAllocationBinding> bindings,
    EngineGpuAllocationOperations& operations) {
  if (bindings.empty())
    return Status::InvalidArgument("GPU allocation manifest is empty");
  std::set<std::uint64_t> allocations;
  std::set<std::string> owner_resources;
  for (const auto& binding : bindings) {
    if (binding.allocation_identity == 0 ||
        zero_allocation_digest(binding.owner_identity) ||
        zero_allocation_digest(binding.resource_identity) ||
        zero_allocation_digest(binding.physical_gpu_identity) ||
        binding.allocated_bytes == 0 ||
        static_cast<std::uint32_t>(binding.state) >= 3 ||
        !allocations.insert(binding.allocation_identity).second ||
        !owner_resources
             .insert(binding.owner_identity.hex() +
                     binding.resource_identity.hex())
             .second) {
      return Status::InvalidArgument("GPU allocation manifest is invalid");
    }
  }
  return EngineGpuAllocationInventory(
      std::vector<EngineGpuAllocationBinding>(bindings.begin(), bindings.end()),
      operations);
}

Result<std::vector<EngineOwnedResourceManifestRecord>>
EngineGpuAllocationInventory::capture(EngineOwnedResourceKind kind) {
  if (kind != EngineOwnedResourceKind::kGpuAllocation)
    return Status::InvalidArgument("GPU allocation inventory kind is invalid");
  std::vector<std::uint64_t> identities;
  identities.reserve(bindings_.size());
  for (const auto& binding : bindings_)
    identities.push_back(binding.allocation_identity);
  auto observations = operations_->capture_allocations(identities);
  if (!observations.ok()) return observations.status();
  if (observations->size() != bindings_.size())
    return Status::FailedPrecondition(
        "GPU allocation snapshot cardinality drifted");
  std::vector<EngineOwnedResourceManifestRecord> records;
  records.reserve(bindings_.size());
  for (std::size_t index = 0; index < bindings_.size(); ++index) {
    const auto& binding = bindings_[index];
    const auto& observation = observations->at(index);
    if (observation.allocation_identity != binding.allocation_identity)
      return Status::FailedPrecondition("GPU allocation identity drifted");
    if (!observation.owner_counter_visible)
      return Status::Unavailable("GPU allocation owner counter is unavailable");
    if (!observation.allocated) {
      if (observation.allocated_bytes != 0)
        return Status::FailedPrecondition(
            "released GPU allocation retained bytes");
      continue;
    }
    if (observation.physical_gpu_identity !=
            binding.physical_gpu_identity ||
        observation.allocated_bytes != binding.allocated_bytes) {
      return Status::FailedPrecondition("GPU allocation owner drifted");
    }
    records.push_back({EngineOwnedResourceKind::kGpuAllocation,
                       binding.owner_identity, binding.resource_identity,
                       observation.allocated_bytes, 1, binding.state});
  }
  return records;
}
}  // namespace pih
