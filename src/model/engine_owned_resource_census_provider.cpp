#include "pih/model/engine_owned_resource_census_provider.h"

#include <array>

namespace pih {

Result<EngineOwnedResourceCensusProvider>
EngineOwnedResourceCensusProvider::Create(
    EngineOwnedResourceKind kind, EngineOwnedResourceCensusSource& source) {
  const std::array<EngineOwnedResourceCensusRecord, 0> empty{};
  auto validation = engine_owned_resource_census_digest(kind, empty);
  if (!validation.ok()) return validation.status();
  return EngineOwnedResourceCensusProvider(kind, source);
}

Result<Sha256Digest> EngineOwnedResourceCensusProvider::observe() {
  auto census = source_->capture();
  if (!census.ok()) return census.status();

  std::vector<EngineOwnedResourceCensusRecord> records;
  records.reserve(census->size());
  for (const auto& identity : *census) records.push_back({identity});
  return engine_owned_resource_census_digest(kind_, records);
}

}  // namespace pih
