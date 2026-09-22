#pragma once

#include <atomic>
#include <cstdint>
#include <string_view>

#include "pih/core/result.h"
#include "pih/model/deepseek_engine_artifact_poller.h"

namespace pih {

enum class ArtifactLeaseEpochState : std::uint8_t {
  kBootstrapping = 0,
  kHealthy = 1,
  kFailed = 2,
};

enum class ArtifactLeaseEpochFailure : std::uint8_t {
  kNone = 0,
  kLeaseIntegrity = 1,
  kMappingFault = 2,
  kWorkerLost = 3,
};

class DeepSeekArtifactEpochGuard final {
 public:
  static Result<DeepSeekArtifactEpochGuard> Create(
      std::uint64_t epoch, std::uint32_t poll_interval_ms,
      std::uint32_t world_size);

  DeepSeekArtifactEpochGuard(const DeepSeekArtifactEpochGuard&) = delete;
  DeepSeekArtifactEpochGuard& operator=(
      const DeepSeekArtifactEpochGuard&) = delete;
  DeepSeekArtifactEpochGuard(DeepSeekArtifactEpochGuard&& other) noexcept;
  DeepSeekArtifactEpochGuard& operator=(
      DeepSeekArtifactEpochGuard&& other) noexcept;

  Status mark_ready();
  Status poll(DeepSeekEngineArtifactPoller& poller);
  Status report_mapping_fault(std::uint32_t rank);
  Status report_worker_lost(std::uint32_t rank);
  Status report_lease_integrity_failure();

  [[nodiscard]] ArtifactLeaseEpochState state() const noexcept {
    return state_.load(std::memory_order_acquire);
  }
  [[nodiscard]] ArtifactLeaseEpochFailure failure() const noexcept {
    return failure_.load(std::memory_order_acquire);
  }
  [[nodiscard]] bool admission_allowed() const noexcept {
    return state() == ArtifactLeaseEpochState::kHealthy;
  }
  [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
  [[nodiscard]] std::uint32_t poll_interval_ms() const noexcept {
    return poll_interval_ms_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] std::string_view public_error() const noexcept {
    return state() == ArtifactLeaseEpochState::kFailed
               ? "engine_artifact_integrity_failed"
               : std::string_view{};
  }

 private:
  DeepSeekArtifactEpochGuard(std::uint64_t epoch,
                             std::uint32_t poll_interval_ms,
                             std::uint32_t world_size)
      : epoch_(epoch),
        poll_interval_ms_(poll_interval_ms),
        world_size_(world_size) {}
  void fail(ArtifactLeaseEpochFailure failure) noexcept;

  std::uint64_t epoch_ = 0;
  std::uint32_t poll_interval_ms_ = 0;
  std::uint32_t world_size_ = 0;
  std::atomic<ArtifactLeaseEpochState> state_{
      ArtifactLeaseEpochState::kBootstrapping};
  std::atomic<ArtifactLeaseEpochFailure> failure_{
      ArtifactLeaseEpochFailure::kNone};
};

}  // namespace pih
