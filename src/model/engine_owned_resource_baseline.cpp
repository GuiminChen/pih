#include "pih/model/engine_owned_resource_baseline.h"

namespace pih {

Result<EngineOwnedResourceBaselineGate>
EngineOwnedResourceBaselineGate::Create(
    std::uint64_t engine_generation, Sha256Digest allocation_lease_digest,
    std::span<const EngineOwnedResourceBaselineEntry> baselines) {
  constexpr std::size_t kResourceKinds = 6;
  if (engine_generation == 0 || allocation_lease_digest == Sha256Digest{} ||
      baselines.size() != kResourceKinds)
    return Status::InvalidArgument("engine owned-resource baseline is invalid");
  for (std::size_t index = 0; index < baselines.size(); ++index) {
    if (baselines[index].kind !=
            static_cast<EngineOwnedResourceKind>(index) ||
        baselines[index].baseline_digest == Sha256Digest{})
      return Status::InvalidArgument(
          "engine owned-resource baseline manifest is not canonical");
  }
  return EngineOwnedResourceBaselineGate(
      engine_generation, allocation_lease_digest,
      std::vector<EngineOwnedResourceBaselineEntry>(baselines.begin(),
                                                    baselines.end()));
}

Result<EngineOwnedResourceBaselineState>
EngineOwnedResourceBaselineGate::accept(
    const EngineOwnedResourceBaselineReceipt& receipt) {
  if (poisoned_)
    return Status::FailedPrecondition(
        "engine owned-resource baseline gate is poisoned");
  if (receipt.engine_generation != generation_ ||
      receipt.allocation_lease_digest != lease_digest_ ||
      receipt.sample_identity == 0 ||
      receipt.sample_identity != last_sample_identity_ + 1 ||
      receipt.sample_completed_ns < receipt.sample_started_ns ||
      receipt.sample_started_ns < last_sample_completed_ns_ ||
      receipt.resources.size() != baselines_.size()) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine owned-resource baseline receipt drifted");
  }
  bool unknown = false;
  bool drifted = false;
  for (std::size_t index = 0; index < baselines_.size(); ++index) {
    const auto& observation = receipt.resources[index];
    if (observation.kind != baselines_[index].kind ||
        (observation.visibility_complete &&
         observation.observed_digest == Sha256Digest{})) {
      poisoned_ = true;
      return Status::FailedPrecondition(
          "engine owned-resource observation is not canonical");
    }
    if (!observation.visibility_complete)
      unknown = true;
    else if (observation.observed_digest != baselines_[index].baseline_digest)
      drifted = true;
    visibility_[index] = observation.visibility_complete;
    at_baseline_[index] = observation.visibility_complete &&
                          observation.observed_digest ==
                              baselines_[index].baseline_digest;
  }
  last_sample_identity_ = receipt.sample_identity;
  last_sample_completed_ns_ = receipt.sample_completed_ns;
  state_ = unknown ? EngineOwnedResourceBaselineState::kUnknown
                   : drifted ? EngineOwnedResourceBaselineState::kDrifted
                             : EngineOwnedResourceBaselineState::kBaseline;
  return state_;
}

bool EngineOwnedResourceBaselineGate::visibility_complete(
    EngineOwnedResourceKind kind) const noexcept {
  const auto index = static_cast<std::size_t>(kind);
  return !poisoned_ && index < visibility_.size() && visibility_[index];
}

bool EngineOwnedResourceBaselineGate::at_baseline(
    EngineOwnedResourceKind kind) const noexcept {
  const auto index = static_cast<std::size_t>(kind);
  return !poisoned_ && index < at_baseline_.size() && at_baseline_[index];
}

bool EngineOwnedResourceBaselineGate::bound_to(
    std::uint64_t generation, const Sha256Digest& lease_digest) const noexcept {
  return generation == generation_ && lease_digest == lease_digest_;
}

}  // namespace pih
