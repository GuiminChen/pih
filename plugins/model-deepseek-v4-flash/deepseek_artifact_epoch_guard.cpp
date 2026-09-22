#include "pih/model/deepseek_artifact_epoch_guard.h"

namespace pih {

Result<DeepSeekArtifactEpochGuard> DeepSeekArtifactEpochGuard::Create(
    std::uint64_t epoch, std::uint32_t poll_interval_ms,
    std::uint32_t world_size) {
  if (epoch == 0 || poll_interval_ms == 0 || world_size == 0 ||
      world_size > 4) {
    return Status::InvalidArgument(
        "DeepSeek artifact epoch guard configuration is invalid");
  }
  return DeepSeekArtifactEpochGuard(epoch, poll_interval_ms, world_size);
}

DeepSeekArtifactEpochGuard::DeepSeekArtifactEpochGuard(
    DeepSeekArtifactEpochGuard&& other) noexcept
    : epoch_(other.epoch_),
      poll_interval_ms_(other.poll_interval_ms_),
      world_size_(other.world_size_),
      state_(other.state()),
      failure_(other.failure()) {}

DeepSeekArtifactEpochGuard& DeepSeekArtifactEpochGuard::operator=(
    DeepSeekArtifactEpochGuard&& other) noexcept {
  if (this != &other) {
    epoch_ = other.epoch_;
    poll_interval_ms_ = other.poll_interval_ms_;
    world_size_ = other.world_size_;
    failure_.store(other.failure(), std::memory_order_relaxed);
    state_.store(other.state(), std::memory_order_release);
  }
  return *this;
}

Status DeepSeekArtifactEpochGuard::mark_ready() {
  auto expected = ArtifactLeaseEpochState::kBootstrapping;
  if (!state_.compare_exchange_strong(expected,
                                      ArtifactLeaseEpochState::kHealthy,
                                      std::memory_order_acq_rel)) {
    return Status::FailedPrecondition(
        "DeepSeek artifact epoch cannot transition to ready");
  }
  return Status::Ok();
}

void DeepSeekArtifactEpochGuard::fail(
    ArtifactLeaseEpochFailure failure) noexcept {
  auto expected = ArtifactLeaseEpochFailure::kNone;
  failure_.compare_exchange_strong(expected, failure,
                                   std::memory_order_acq_rel);
  state_.store(ArtifactLeaseEpochState::kFailed,
               std::memory_order_release);
}


Status DeepSeekArtifactEpochGuard::poll(DeepSeekEngineArtifactPoller& poller) {
  if (state() != ArtifactLeaseEpochState::kHealthy)
    return Status::FailedPrecondition("DeepSeek artifact epoch is not healthy");
  if (poller.world_size() != world_size_)
    return report_lease_integrity_failure();
  try {
    const auto status = poller.poll();
    if (!status.ok()) {
      fail(ArtifactLeaseEpochFailure::kLeaseIntegrity);
      return status;
    }
  } catch (...) {
    fail(ArtifactLeaseEpochFailure::kLeaseIntegrity);
    return Status::Internal("DeepSeek artifact capability poll threw an exception");
  }
  // Another reporter may have invalidated the epoch during the storage call.
  return admission_allowed() ? Status::Ok()
      : Status::FailedPrecondition("DeepSeek artifact epoch failed during poll");
}

Status DeepSeekArtifactEpochGuard::report_mapping_fault(std::uint32_t rank) {
  if (rank >= world_size_)
    return Status::InvalidArgument("DeepSeek mapping fault rank is invalid");
  fail(ArtifactLeaseEpochFailure::kMappingFault);
  return Status::Ok();
}

Status DeepSeekArtifactEpochGuard::report_worker_lost(std::uint32_t rank) {
  if (rank >= world_size_)
    return Status::InvalidArgument("DeepSeek lost worker rank is invalid");
  fail(ArtifactLeaseEpochFailure::kWorkerLost);
  return Status::Ok();
}

Status DeepSeekArtifactEpochGuard::report_lease_integrity_failure() {
  fail(ArtifactLeaseEpochFailure::kLeaseIntegrity);
  return Status::FailedPrecondition(
      "DeepSeek artifact lease integrity failed");
}

}  // namespace pih
