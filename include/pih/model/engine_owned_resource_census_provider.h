#pragma once

#include <vector>

#include "pih/model/engine_owned_resource_census_digest.h"
#include "pih/model/engine_owned_resource_provider_set.h"

namespace pih {

using EngineOwnedResourceCensus = std::vector<std::vector<std::byte>>;

class EngineOwnedResourceCensusSource {
 public:
  virtual ~EngineOwnedResourceCensusSource() = default;
  virtual Result<EngineOwnedResourceCensus> capture() = 0;
};

class EngineOwnedResourceCensusProvider final
    : public EngineOwnedResourceKindProvider {
 public:
  // The source is not owned and must outlive this provider.
  static Result<EngineOwnedResourceCensusProvider> Create(
      EngineOwnedResourceKind kind, EngineOwnedResourceCensusSource& source);

  Result<Sha256Digest> observe() override;

 private:
  EngineOwnedResourceCensusProvider(
      EngineOwnedResourceKind kind,
      EngineOwnedResourceCensusSource& source) noexcept
      : kind_(kind), source_(&source) {}

  EngineOwnedResourceKind kind_ = EngineOwnedResourceKind::kShm;
  EngineOwnedResourceCensusSource* source_ = nullptr;
};

}  // namespace pih
