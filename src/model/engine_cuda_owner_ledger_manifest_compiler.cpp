#include "pih/model/engine_cuda_owner_ledger_manifest_compiler.h"

#include <set>
#include <string>

#include "pih/model/engine_owned_resource_census_digest.h"

namespace pih {
namespace {

bool zero_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes)
    if (value != std::byte{0}) return false;
  return true;
}

Status validate_bindings(
    std::span<const EnginePinnedMemoryBinding> pinned_bindings,
    std::span<const EngineGpuAllocationBinding> allocation_bindings) {
  // A full-resident engine has no engine-owned pinned extent.  Its absence is
  // represented by the caller's explicit zero-pinned sentinel, whereas every
  // census still requires at least one device allocation owner.
  if (allocation_bindings.empty())
    return Status::InvalidArgument("CUDA allocation owner manifest is empty");
  std::set<std::uint64_t> counter_identities;
  std::set<std::string> semantic_identities;
  for (const auto& binding : pinned_bindings) {
    if (binding.registration_identity == 0 ||
        zero_digest(binding.owner_identity) ||
        zero_digest(binding.resource_identity) ||
        zero_digest(binding.physical_gpu_identity) || binding.numa_node < 0 ||
        binding.registered_bytes == 0 ||
        static_cast<std::uint32_t>(binding.state) >= 3 ||
        !counter_identities.insert(binding.registration_identity).second ||
        !semantic_identities
             .insert(binding.owner_identity.hex() +
                     binding.resource_identity.hex())
             .second)
      return Status::InvalidArgument("pinned CUDA owner binding is invalid");
  }
  for (const auto& binding : allocation_bindings) {
    if (binding.allocation_identity == 0 ||
        zero_digest(binding.owner_identity) ||
        zero_digest(binding.resource_identity) ||
        zero_digest(binding.physical_gpu_identity) ||
        binding.allocated_bytes == 0 ||
        static_cast<std::uint32_t>(binding.state) >= 3 ||
        !counter_identities.insert(binding.allocation_identity).second ||
        !semantic_identities
             .insert(binding.owner_identity.hex() +
                     binding.resource_identity.hex())
             .second)
      return Status::InvalidArgument("GPU allocation binding is invalid");
  }
  return Status::Ok();
}

Result<Sha256Digest> manifest_records_digest(
    EngineOwnedResourceKind kind,
    std::span<const EngineOwnedResourceManifestRecord> records) {
  std::vector<std::vector<std::byte>> encoded;
  encoded.reserve(records.size());
  for (const auto& record : records) {
    auto value = encode_engine_owned_resource_manifest_record(record);
    if (!value.ok()) return value.status();
    encoded.push_back(std::move(*value));
  }
  std::vector<EngineOwnedResourceCensusRecord> census;
  census.reserve(encoded.size());
  for (const auto& value : encoded) census.push_back({value});
  return engine_owned_resource_census_digest(kind, census);
}

}  // namespace

Result<EngineCudaOwnerLedgerManifestSample>
compile_engine_cuda_owner_ledger_manifest(
    const EngineCudaOwnerLedgerSnapshot& snapshot,
    std::span<const EnginePinnedMemoryBinding> pinned_bindings,
    std::span<const EngineGpuAllocationBinding> allocation_bindings) {
  auto binding_status = validate_bindings(pinned_bindings,
                                          allocation_bindings);
  if (!binding_status.ok()) return binding_status;
  if (snapshot.sample_identity == 0 || snapshot.sample_started_ns == 0 ||
      snapshot.sample_completed_ns < snapshot.sample_started_ns)
    return Status::FailedPrecondition("CUDA owner sample window is invalid");
  if (snapshot.pinned.size() != pinned_bindings.size() ||
      snapshot.allocations.size() != allocation_bindings.size())
    return Status::FailedPrecondition(
        "CUDA owner sample cardinality drifted");

  EngineCudaOwnerLedgerManifestSample result{
      snapshot.sample_identity, snapshot.sample_started_ns,
      snapshot.sample_completed_ns, {}, {}, {}, {}};
  result.pinned_records.reserve(pinned_bindings.size());
  result.allocation_records.reserve(allocation_bindings.size());
  for (std::size_t index = 0; index < pinned_bindings.size(); ++index) {
    const auto& binding = pinned_bindings[index];
    const auto& observation = snapshot.pinned[index];
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
    if (observation.physical_gpu_identity != binding.physical_gpu_identity ||
        observation.numa_node != binding.numa_node ||
        observation.registered_bytes != binding.registered_bytes)
      return Status::FailedPrecondition("pinned memory topology drifted");
    result.pinned_records.push_back(
        {EngineOwnedResourceKind::kPinnedMemory, binding.owner_identity,
         binding.resource_identity, observation.registered_bytes, 1,
         binding.state});
  }
  for (std::size_t index = 0; index < allocation_bindings.size(); ++index) {
    const auto& binding = allocation_bindings[index];
    const auto& observation = snapshot.allocations[index];
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
    if (observation.physical_gpu_identity != binding.physical_gpu_identity ||
        observation.allocated_bytes != binding.allocated_bytes)
      return Status::FailedPrecondition("GPU allocation owner drifted");
    result.allocation_records.push_back(
        {EngineOwnedResourceKind::kGpuAllocation, binding.owner_identity,
         binding.resource_identity, observation.allocated_bytes, 1,
         binding.state});
  }
  Sha256Digest pinned_digest{};
  if (!pinned_bindings.empty()) {
    auto compiled_pinned_digest = manifest_records_digest(
        EngineOwnedResourceKind::kPinnedMemory, result.pinned_records);
    if (!compiled_pinned_digest.ok()) return compiled_pinned_digest.status();
    pinned_digest = *compiled_pinned_digest;
  }
  auto allocation_digest = manifest_records_digest(
      EngineOwnedResourceKind::kGpuAllocation, result.allocation_records);
  if (!allocation_digest.ok()) return allocation_digest.status();
  result.pinned_census_digest = pinned_digest;
  result.allocation_census_digest = *allocation_digest;
  return result;
}

}  // namespace pih
