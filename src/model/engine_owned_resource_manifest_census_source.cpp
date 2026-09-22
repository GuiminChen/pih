#include "pih/model/engine_owned_resource_manifest_census_source.h"

namespace pih {

Result<EngineOwnedResourceManifestCensusSource>
EngineOwnedResourceManifestCensusSource::Create(
    EngineOwnedResourceKind kind,
    EngineOwnedResourceManifestInventory& inventory) {
  if (static_cast<std::uint32_t>(kind) >= 6) {
    return Status::InvalidArgument(
        "engine owned-resource manifest census kind is invalid");
  }
  return EngineOwnedResourceManifestCensusSource(kind, inventory);
}

Result<EngineOwnedResourceCensus>
EngineOwnedResourceManifestCensusSource::capture() {
  auto observations = inventory_->capture(kind_);
  if (!observations.ok()) return observations.status();

  EngineOwnedResourceCensus census;
  census.reserve(observations->size());
  for (const auto& observation : *observations) {
    if (observation.kind != kind_) {
      return Status::FailedPrecondition(
          "engine owned-resource manifest inventory kind drifted");
    }
    auto encoded = encode_engine_owned_resource_manifest_record(observation);
    if (!encoded.ok()) return encoded.status();
    census.push_back(std::move(*encoded));
  }
  return census;
}

}  // namespace pih
