#include "pih/model/engine_cuda_owner_ledger_operations.h"

#include <algorithm>
#include <set>

namespace pih {
namespace {

template <typename Observation, typename Identity>
bool exact_identity_order(std::span<const Observation> observations,
                          std::span<const std::uint64_t> identities,
                          Identity identity) {
  if (observations.size() != identities.size()) return false;
  for (std::size_t index = 0; index < identities.size(); ++index) {
    if (identity(observations[index]) != identities[index]) return false;
  }
  return true;
}

bool exact_request(std::span<const std::uint64_t> request,
                   const std::vector<std::uint64_t>& configured) {
  return std::equal(request.begin(), request.end(), configured.begin(),
                    configured.end());
}

}  // namespace

Result<std::unique_ptr<EngineCudaOwnerLedgerOperations>>
EngineCudaOwnerLedgerOperations::Create(
    std::span<const std::uint64_t> pinned_identities,
    std::span<const std::uint64_t> allocation_identities,
    EngineCudaOwnerLedgerBackend& backend) {
  if (pinned_identities.empty() || allocation_identities.empty())
    return Status::InvalidArgument("CUDA owner ledger identities are empty");
  std::set<std::uint64_t> identities;
  for (const auto identity : pinned_identities) {
    if (identity == 0 || !identities.insert(identity).second)
      return Status::InvalidArgument("CUDA owner ledger identity is invalid");
  }
  for (const auto identity : allocation_identities) {
    if (identity == 0 || !identities.insert(identity).second)
      return Status::InvalidArgument("CUDA owner ledger identity is invalid");
  }
  return std::unique_ptr<EngineCudaOwnerLedgerOperations>(
      new EngineCudaOwnerLedgerOperations(
      std::vector<std::uint64_t>(pinned_identities.begin(),
                                 pinned_identities.end()),
      std::vector<std::uint64_t>(allocation_identities.begin(),
                                 allocation_identities.end()),
      backend));
}

Status EngineCudaOwnerLedgerOperations::poison_locked(const char* message) {
  poisoned_ = true;
  return Status::FailedPrecondition(message);
}

Result<EngineCudaOwnerLedgerSnapshot>
EngineCudaOwnerLedgerOperations::sample_locked() {
  if (poisoned_)
    return Status::FailedPrecondition("CUDA owner ledger is poisoned");
  auto snapshot = backend_->capture(pinned_identities_, allocation_identities_);
  if (!snapshot.ok()) {
    if (snapshot.status().code() == StatusCode::kUnavailable)
      return snapshot.status();
    poisoned_ = true;
    return snapshot.status();
  }
  if (snapshot->sample_identity == 0 ||
      snapshot->sample_completed_ns < snapshot->sample_started_ns ||
      (last_sample_identity_ != 0 &&
       snapshot->sample_identity <= last_sample_identity_) ||
      (last_sample_identity_ != 0 &&
       snapshot->sample_started_ns < last_sample_completed_ns_)) {
    return poison_locked("CUDA owner ledger sample window drifted");
  }
  if (!exact_identity_order(
          std::span<const EnginePinnedMemoryObservation>(snapshot->pinned),
          pinned_identities_, [](const auto& value) {
            return value.registration_identity;
          }) ||
      !exact_identity_order(
          std::span<const EngineGpuAllocationObservation>(
              snapshot->allocations),
          allocation_identities_, [](const auto& value) {
            return value.allocation_identity;
          })) {
    return poison_locked("CUDA owner ledger membership drifted");
  }
  last_sample_identity_ = snapshot->sample_identity;
  last_sample_completed_ns_ = snapshot->sample_completed_ns;
  return snapshot;
}

Result<std::vector<EnginePinnedMemoryObservation>>
EngineCudaOwnerLedgerOperations::capture_pinned(
    std::span<const std::uint64_t> registration_identities) {
  std::lock_guard lock(mutex_);
  if (!exact_request(registration_identities, pinned_identities_))
    return Status::InvalidArgument("CUDA pinned owner request drifted");
  auto snapshot = sample_locked();
  if (!snapshot.ok()) return snapshot.status();
  return std::move(snapshot->pinned);
}

Result<std::vector<EngineGpuAllocationObservation>>
EngineCudaOwnerLedgerOperations::capture_allocations(
    std::span<const std::uint64_t> allocation_identities) {
  std::lock_guard lock(mutex_);
  if (!exact_request(allocation_identities, allocation_identities_))
    return Status::InvalidArgument("CUDA allocation owner request drifted");
  auto snapshot = sample_locked();
  if (!snapshot.ok()) return snapshot.status();
  return std::move(snapshot->allocations);
}

}  // namespace pih
