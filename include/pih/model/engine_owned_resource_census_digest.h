#pragma once

#include <span>

#include "pih/model/engine_owned_resource_baseline.h"

namespace pih {

struct EngineOwnedResourceCensusRecord final {
  std::span<const std::byte> identity;
};

// Produces an order-independent digest of a complete, visible resource census.
// Record identities are adapter-defined canonical binary values. Their backing
// storage only needs to remain alive for the duration of this call.
Result<Sha256Digest> engine_owned_resource_census_digest(
    EngineOwnedResourceKind kind,
    std::span<const EngineOwnedResourceCensusRecord> records);

}  // namespace pih
