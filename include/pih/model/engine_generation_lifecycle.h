#pragma once

#include <cstdint>
#include <deque>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

enum class EngineGenerationState {
  kLeased,
  kStarting,
  kReady,
  kDraining,
  kStopping,
  kFailed,
  kDead,
  kQuiescenceCheck,
  kBackoff,
  kQuarantined,
};

enum class EngineQuiescenceState { kPending, kVerified, kUnknown };

struct EngineGenerationQuiescenceObservation final {
  bool visibility_complete = false;
  bool cgroup_domain_empty = false;
  bool pidfds_reaped = false;
  bool target_gpu_processes_absent = false;
  bool shm_baseline = false;
  bool network_baseline = false;
  bool pinned_memory_baseline = false;
  bool listener_baseline = false;
  bool artifact_baseline = false;
  bool gpu_allocation_baseline = false;
};

struct EngineGenerationQuiescenceReceipt final {
  std::uint64_t generation = 0;
  Sha256Digest allocation_lease_digest{};
  std::uint64_t sample_identity = 0;
  std::uint64_t sample_started_ns = 0;
  std::uint64_t sample_completed_ns = 0;
  EngineGenerationQuiescenceObservation observation;
};

struct EngineRestartPolicy final {
  std::uint64_t backoff_initial_ns = 0;
  std::uint64_t backoff_max_ns = 0;
  std::uint64_t restart_window_ns = 0;
  std::uint32_t restart_burst_max = 0;
};

class EngineGenerationLifecycle final {
 public:
  static Result<EngineGenerationLifecycle> Create(
      std::uint64_t generation, Sha256Digest allocation_lease_digest,
      EngineRestartPolicy policy);

  Status begin_start();
  Status publish_ready();
  Status begin_drain();
  Status abort_start();
  Status begin_stop();
  Status fail_generation(bool automatic_restart_eligible);
  Status controller_reaped();
  Status begin_quiescence_check();
  Result<EngineQuiescenceState> observe_quiescence(
      const EngineGenerationQuiescenceReceipt& receipt,
      std::uint64_t now_ns);
  Status quarantine() noexcept;
  Status finish_backoff(std::uint64_t now_ns, std::uint64_t next_generation);

  [[nodiscard]] EngineGenerationState state() const noexcept { return state_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] std::uint64_t retry_not_before_ns() const noexcept {
    return retry_not_before_ns_;
  }

 private:
  EngineGenerationLifecycle(std::uint64_t generation,
                            Sha256Digest allocation_lease_digest,
                            EngineRestartPolicy policy) noexcept
      : generation_(generation),
        allocation_lease_digest_(allocation_lease_digest), policy_(policy) {}
  Status require(EngineGenerationState expected, EngineGenerationState next);
  std::uint64_t generation_ = 0;
  Sha256Digest allocation_lease_digest_{};
  EngineRestartPolicy policy_{};
  EngineGenerationState state_ = EngineGenerationState::kLeased;
  std::deque<std::uint64_t> attempts_;
  std::uint64_t retry_not_before_ns_ = 0;
  std::uint64_t last_restart_observation_ns_ = 0;
  std::uint64_t last_quiescence_sample_identity_ = 0;
  std::uint64_t last_quiescence_sample_completed_ns_ = 0;
  bool automatic_restart_eligible_ = false;
};

}  // namespace pih
