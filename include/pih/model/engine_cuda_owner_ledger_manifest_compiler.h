#pragma once

#include "pih/model/engine_cuda_owner_ledger_operations.h"

namespace pih {

struct EngineCudaOwnerLedgerManifestSample final {
  std::uint64_t sample_identity = 0;
  std::uint64_t sample_started_ns = 0;
  std::uint64_t sample_completed_ns = 0;
  Sha256Digest pinned_census_digest{};
  Sha256Digest allocation_census_digest{};
  std::vector<EngineOwnedResourceManifestRecord> pinned_records;
  std::vector<EngineOwnedResourceManifestRecord> allocation_records;
};

Result<EngineCudaOwnerLedgerManifestSample>
compile_engine_cuda_owner_ledger_manifest(
    const EngineCudaOwnerLedgerSnapshot& snapshot,
    std::span<const EnginePinnedMemoryBinding> pinned_bindings,
    std::span<const EngineGpuAllocationBinding> allocation_bindings);

}  // namespace pih
