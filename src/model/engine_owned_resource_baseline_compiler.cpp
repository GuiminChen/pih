#include "pih/model/engine_owned_resource_baseline_compiler.h"

#include <limits>

namespace pih {

Result<EngineOwnedResourceBaselineCompiler>
EngineOwnedResourceBaselineCompiler::Create(
    std::uint64_t engine_generation, Sha256Digest allocation_lease_digest,
    std::span<const EngineOwnedResourceBaselineEntry> baselines,
    EngineOwnedResourceProvider& provider) {
  auto validator = EngineOwnedResourceBaselineGate::Create(
      engine_generation, allocation_lease_digest, baselines);
  if (!validator.ok()) return validator.status();
  return EngineOwnedResourceBaselineCompiler(
      engine_generation, allocation_lease_digest,
      std::vector<EngineOwnedResourceBaselineEntry>(baselines.begin(),
                                                    baselines.end()),
      provider, std::move(*validator));
}

Result<EngineOwnedResourceBaselineReceipt>
EngineOwnedResourceBaselineCompiler::capture(
    std::uint64_t sample_started_ns, std::uint64_t sample_completed_ns) {
  if (poisoned_)
    return Status::FailedPrecondition(
        "engine owned-resource compiler is poisoned");
  if (sample_completed_ns < sample_started_ns ||
      sample_started_ns < last_sample_completed_ns_ ||
      next_sample_identity_ == std::numeric_limits<std::uint64_t>::max()) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine owned-resource sample time drifted");
  }
  std::vector<EngineOwnedResourceObservation> observations;
  observations.reserve(baselines_.size());
  for (const auto& baseline : baselines_) {
    auto observed = provider_->observe(baseline.kind);
    if (!observed.ok()) {
      if (observed.status().code() != StatusCode::kUnavailable) {
        poisoned_ = true;
        return observed.status();
      }
      observations.push_back({baseline.kind, false, {}});
    } else {
      observations.push_back({baseline.kind, true, *observed});
    }
  }
  EngineOwnedResourceBaselineReceipt receipt{
      generation_, lease_digest_, next_sample_identity_, sample_started_ns,
      sample_completed_ns, std::move(observations)};
  auto validated = validator_.accept(receipt);
  if (!validated.ok()) {
    poisoned_ = true;
    return validated.status();
  }
  ++next_sample_identity_;
  last_sample_completed_ns_ = sample_completed_ns;
  return receipt;
}

}  // namespace pih
