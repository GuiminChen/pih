#pragma once

#include "pih/model/engine_owned_resource_census_provider.h"
#include "pih/model/engine_owned_resource_manifest_record.h"

namespace pih {

class EngineOwnedResourceManifestInventory {
 public:
  virtual ~EngineOwnedResourceManifestInventory() = default;
  virtual Result<std::vector<EngineOwnedResourceManifestRecord>> capture(
      EngineOwnedResourceKind kind) = 0;
};

class EngineOwnedResourceManifestCensusSource final
    : public EngineOwnedResourceCensusSource {
 public:
  // The inventory is not owned and must outlive this source.
  static Result<EngineOwnedResourceManifestCensusSource> Create(
      EngineOwnedResourceKind kind,
      EngineOwnedResourceManifestInventory& inventory);

  Result<EngineOwnedResourceCensus> capture() override;

 private:
  EngineOwnedResourceManifestCensusSource(
      EngineOwnedResourceKind kind,
      EngineOwnedResourceManifestInventory& inventory) noexcept
      : kind_(kind), inventory_(&inventory) {}

  EngineOwnedResourceKind kind_ = EngineOwnedResourceKind::kShm;
  EngineOwnedResourceManifestInventory* inventory_ = nullptr;
};

}  // namespace pih
