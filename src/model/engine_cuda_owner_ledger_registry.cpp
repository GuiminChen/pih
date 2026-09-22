#include "pih/model/engine_cuda_owner_ledger_registry.h"

#include <limits>

namespace pih {
namespace {

bool zero_registry_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes)
    if (value != std::byte{0}) return false;
  return true;
}

}  // namespace

Status EngineCudaOwnerLedgerRegistry::register_pinned(
    std::uint64_t registration_identity,
    Sha256Digest physical_gpu_identity, std::int32_t numa_node,
    std::uint64_t registered_bytes) {
  if (registration_identity == 0 ||
      zero_registry_digest(physical_gpu_identity) || numa_node < 0 ||
      registered_bytes == 0)
    return Status::InvalidArgument("pinned owner registration is invalid");
  std::lock_guard lock(mutex_);
  if (poisoned_)
    return Status::FailedPrecondition("CUDA owner registry is poisoned");
  if (allocations_.contains(registration_identity))
    return Status::FailedPrecondition("CUDA owner identity crosses categories");
  const auto existing = pinned_.find(registration_identity);
  if (existing != pinned_.end())
    return Status::FailedPrecondition("pinned owner identity was already used");
  pinned_[registration_identity] = EnginePinnedMemoryObservation{
      registration_identity, true, true, physical_gpu_identity, numa_node,
      registered_bytes};
  return Status::Ok();
}

Status EngineCudaOwnerLedgerRegistry::release_pinned(
    std::uint64_t registration_identity) {
  std::lock_guard lock(mutex_);
  if (poisoned_)
    return Status::FailedPrecondition("CUDA owner registry is poisoned");
  const auto existing = pinned_.find(registration_identity);
  if (existing == pinned_.end() || !existing->second.registered)
    return Status::FailedPrecondition("pinned owner is not registered");
  existing->second.registered = false;
  existing->second.registered_bytes = 0;
  return Status::Ok();
}

Status EngineCudaOwnerLedgerRegistry::register_allocation(
    std::uint64_t allocation_identity, Sha256Digest physical_gpu_identity,
    std::uint64_t allocated_bytes) {
  if (allocation_identity == 0 ||
      zero_registry_digest(physical_gpu_identity) || allocated_bytes == 0)
    return Status::InvalidArgument("GPU owner allocation is invalid");
  std::lock_guard lock(mutex_);
  if (poisoned_)
    return Status::FailedPrecondition("CUDA owner registry is poisoned");
  if (pinned_.contains(allocation_identity))
    return Status::FailedPrecondition("CUDA owner identity crosses categories");
  const auto existing = allocations_.find(allocation_identity);
  if (existing != allocations_.end())
    return Status::FailedPrecondition("GPU owner identity was already used");
  allocations_[allocation_identity] = EngineGpuAllocationObservation{
      allocation_identity, true, true, physical_gpu_identity, allocated_bytes};
  return Status::Ok();
}

Status EngineCudaOwnerLedgerRegistry::release_allocation(
    std::uint64_t allocation_identity) {
  std::lock_guard lock(mutex_);
  if (poisoned_)
    return Status::FailedPrecondition("CUDA owner registry is poisoned");
  const auto existing = allocations_.find(allocation_identity);
  if (existing == allocations_.end() || !existing->second.allocated)
    return Status::FailedPrecondition("GPU owner is not allocated");
  existing->second.allocated = false;
  existing->second.allocated_bytes = 0;
  return Status::Ok();
}

Result<EngineCudaOwnerLedgerSnapshot> EngineCudaOwnerLedgerRegistry::capture(
    std::span<const std::uint64_t> pinned_identities,
    std::span<const std::uint64_t> allocation_identities) {
  std::lock_guard lock(mutex_);
  if (poisoned_)
    return Status::FailedPrecondition("CUDA owner registry is poisoned");
  EngineCudaOwnerLedgerSnapshot snapshot;
  snapshot.pinned.reserve(pinned_identities.size());
  snapshot.allocations.reserve(allocation_identities.size());
  for (const auto identity : pinned_identities) {
    const auto found = pinned_.find(identity);
    if (found == pinned_.end())
      return Status::Unavailable("pinned owner is not visible");
    snapshot.pinned.push_back(found->second);
  }
  for (const auto identity : allocation_identities) {
    const auto found = allocations_.find(identity);
    if (found == allocations_.end())
      return Status::Unavailable("GPU owner is not visible");
    snapshot.allocations.push_back(found->second);
  }
  if (next_sample_identity_ == 0 ||
      next_sample_identity_ == std::numeric_limits<std::uint64_t>::max())
    return Status::FailedPrecondition("CUDA owner sample identity exhausted");
  auto started = clock_->monotonic_ns();
  if (!started.ok()) return started.status();
  auto completed = clock_->monotonic_ns();
  if (!completed.ok()) return completed.status();
  if (*completed < *started)
    return Status::FailedPrecondition("CUDA owner clock regressed");
  snapshot.sample_identity = next_sample_identity_++;
  snapshot.sample_started_ns = *started;
  snapshot.sample_completed_ns = *completed;
  return snapshot;
}

void EngineCudaOwnerLedgerRegistry::poison(const char*) noexcept {
  std::lock_guard lock(mutex_);
  poisoned_ = true;
}

}  // namespace pih
