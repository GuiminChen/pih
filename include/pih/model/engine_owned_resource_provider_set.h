#pragma once

#include <array>
#include <span>

#include "pih/model/engine_owned_resource_baseline_compiler.h"

namespace pih {

class EngineOwnedResourceKindProvider {
 public:
  virtual ~EngineOwnedResourceKindProvider() = default;
  virtual Result<Sha256Digest> observe() = 0;
};

struct EngineOwnedResourceProviderBinding final {
  EngineOwnedResourceKind kind = EngineOwnedResourceKind::kShm;
  EngineOwnedResourceKindProvider* provider = nullptr;
};

class EngineOwnedResourceProviderSet final : public EngineOwnedResourceProvider {
 public:
  // Bound providers are not owned and must outlive this set.
  static Result<EngineOwnedResourceProviderSet> Create(
      std::span<const EngineOwnedResourceProviderBinding> bindings);

  Result<Sha256Digest> observe(EngineOwnedResourceKind kind) override;

 private:
  explicit EngineOwnedResourceProviderSet(
      std::array<EngineOwnedResourceKindProvider*, 6> providers) noexcept
      : providers_(providers) {}

  std::array<EngineOwnedResourceKindProvider*, 6> providers_{};
};

}  // namespace pih
