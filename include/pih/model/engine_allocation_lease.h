#pragma once

#include <span>
#include <vector>

#include "pih/core/canonical_hash.h"

namespace pih {

struct EngineAllocationLeaseManifest final {
  Sha256Digest deployment_instance_digest{};
  std::vector<std::uint64_t> sorted_physical_gpu_identities;
};

struct EngineAllocationLeaseExpectation final {
  Sha256Digest lease_digest{};
  std::uint64_t filesystem_identity = 0;
  std::uint64_t file_identity = 0;
};

struct EngineAllocationLeaseObservation final {
  Sha256Digest token_digest{};
  std::uint64_t filesystem_identity = 0;
  std::uint64_t file_identity = 0;
  bool descriptor_open = false;
  bool exclusive_ofd_lock_held = false;
};

Result<Sha256Digest> engine_allocation_lease_digest(
    const EngineAllocationLeaseManifest& manifest);
Status verify_engine_allocation_lease(
    const EngineAllocationLeaseExpectation& expectation,
    const EngineAllocationLeaseObservation& observation);

}  // namespace pih
