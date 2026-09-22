#pragma once

#include "pih/model/engine_owned_resource_baseline.h"

namespace pih {

class EngineOwnedResourceProvider {
 public:
  virtual ~EngineOwnedResourceProvider() = default;
  virtual Result<Sha256Digest> observe(EngineOwnedResourceKind kind) = 0;
};

class EngineOwnedResourceBaselineCompiler final {
 public:
  static Result<EngineOwnedResourceBaselineCompiler> Create(
      std::uint64_t engine_generation, Sha256Digest allocation_lease_digest,
      std::span<const EngineOwnedResourceBaselineEntry> baselines,
      EngineOwnedResourceProvider& provider);
  Result<EngineOwnedResourceBaselineReceipt> capture(
      std::uint64_t sample_started_ns,
      std::uint64_t sample_completed_ns);

 private:
  EngineOwnedResourceBaselineCompiler(
      std::uint64_t generation, Sha256Digest lease_digest,
      std::vector<EngineOwnedResourceBaselineEntry> baselines,
      EngineOwnedResourceProvider& provider,
      EngineOwnedResourceBaselineGate validator) noexcept
      : generation_(generation), lease_digest_(lease_digest),
        baselines_(std::move(baselines)), provider_(&provider),
        validator_(std::move(validator)) {}
  std::uint64_t generation_ = 0;
  Sha256Digest lease_digest_{};
  std::vector<EngineOwnedResourceBaselineEntry> baselines_;
  EngineOwnedResourceProvider* provider_ = nullptr;
  EngineOwnedResourceBaselineGate validator_;
  std::uint64_t next_sample_identity_ = 1;
  std::uint64_t last_sample_completed_ns_ = 0;
  bool poisoned_ = false;
};

}  // namespace pih
