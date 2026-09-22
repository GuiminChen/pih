#include "pih/model/engine_owned_resource_provider_set.h"

namespace pih {
namespace {

constexpr std::size_t kResourceKindCount = 6;

Result<std::size_t> resource_kind_index(EngineOwnedResourceKind kind) {
  const auto index = static_cast<std::size_t>(kind);
  if (index >= kResourceKindCount) {
    return Status::InvalidArgument("invalid engine owned-resource kind");
  }
  return index;
}

}  // namespace

Result<EngineOwnedResourceProviderSet> EngineOwnedResourceProviderSet::Create(
    std::span<const EngineOwnedResourceProviderBinding> bindings) {
  if (bindings.size() != kResourceKindCount) {
    return Status::InvalidArgument(
        "engine owned-resource provider set requires exactly six bindings");
  }

  std::array<EngineOwnedResourceKindProvider*, kResourceKindCount> providers{};
  std::array<bool, kResourceKindCount> seen{};
  for (const auto& binding : bindings) {
    auto index = resource_kind_index(binding.kind);
    if (!index.ok()) return index.status();
    if (binding.provider == nullptr) {
      return Status::InvalidArgument(
          "engine owned-resource provider binding is null");
    }
    if (seen[*index]) {
      return Status::InvalidArgument(
          "engine owned-resource provider binding is duplicated");
    }
    seen[*index] = true;
    providers[*index] = binding.provider;
  }
  return EngineOwnedResourceProviderSet(providers);
}

Result<Sha256Digest> EngineOwnedResourceProviderSet::observe(
    EngineOwnedResourceKind kind) {
  auto index = resource_kind_index(kind);
  if (!index.ok()) return index.status();
  return providers_[*index]->observe();
}

}  // namespace pih
