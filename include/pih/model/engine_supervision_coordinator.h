#pragma once

#include "pih/model/engine_allocation_lease.h"
#include "pih/model/engine_generation_lifecycle.h"
#include "pih/model/engine_progress_watchdog.h"
#include "pih/model/engine_shutdown_controller.h"
#include "pih/model/engine_generation_domain_termination.h"
#include "pih/model/engine_termination_event_gate.h"
#include "pih/model/engine_pidfd_reaping_ledger.h"
#include "pih/model/engine_gpu_process_inventory.h"
#include "pih/model/engine_owned_resource_baseline.h"

namespace pih {

enum class EngineSupervisionFailure : std::uint8_t {
  kNone,
  kAllocationLeaseDrift,
  kRankLoss,
  kProgressWatchdogExpired,
};

class EngineSupervisionCoordinator final {
 public:
  static Result<EngineSupervisionCoordinator> Create(
      std::uint64_t generation, std::uint32_t world_size,
      std::uint64_t watchdog_interval_ns, EngineRestartPolicy restart_policy,
      EngineAllocationLeaseExpectation lease_expectation);
  Status begin_start(const EngineAllocationLeaseObservation& lease,
                     std::uint64_t now_ns);
  Status publish_ready(const EngineAllocationLeaseObservation& lease);
  Status accept_heartbeat(const EngineProgressHeartbeat& heartbeat,
                          std::uint64_t now_ns);
  Status poll(const EngineAllocationLeaseObservation& lease,
              bool all_ranks_healthy, std::uint64_t now_ns);
  Status report_rank_loss();
  Result<EngineShutdownAction> request_termination(
      EngineShutdownController& shutdown, std::uint64_t now_ns);
  Result<EngineShutdownAction> consume_termination_event(
      EngineTerminationEventGate& gate,
      const EngineTerminationEvent& event,
      EngineShutdownController& shutdown, std::uint64_t now_ns);
  Result<EngineShutdownAction> poll_shutdown(
      EngineShutdownController& shutdown, std::uint64_t now_ns,
      bool committed_work_remaining);
  Status mark_shutdown_domain_empty(EngineShutdownController& shutdown);
  Status execute_force_kill(
      EngineShutdownAction action, const EngineShutdownController& shutdown,
      EngineGenerationDomainTermination& termination);
  Result<bool> reconcile_shutdown_domain(
      EngineShutdownController& shutdown,
      EngineGenerationDomainTermination& termination);
  Status controller_reaped();
  Status begin_quiescence_check();
  Result<EngineQuiescenceState> observe_quiescence(
      const EngineGenerationQuiescenceReceipt& receipt,
      const EnginePidfdReapingLedger& pidfd_ledger,
      const EngineGpuProcessInventoryGate& gpu_inventory,
      const EngineOwnedResourceBaselineGate& owned_resources,
      std::uint64_t now_ns);
  Status authorize_restart(
      std::uint64_t now_ns, std::uint64_t next_generation,
      const EngineAllocationLeaseObservation& lease);

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] EngineSupervisionFailure first_failure() const noexcept {
    return first_failure_;
  }
  [[nodiscard]] EngineGenerationState lifecycle_state() const noexcept {
    return lifecycle_.state();
  }
  [[nodiscard]] std::uint64_t retry_not_before_ns() const noexcept {
    return lifecycle_.retry_not_before_ns();
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return lifecycle_.generation();
  }

 private:
  EngineSupervisionCoordinator(
      EngineGenerationLifecycle lifecycle, EngineProgressWatchdog watchdog,
      EngineAllocationLeaseExpectation expectation, std::uint32_t world_size,
      std::uint64_t watchdog_interval_ns) noexcept
      : lifecycle_(std::move(lifecycle)), watchdog_(std::move(watchdog)),
        lease_expectation_(expectation), world_size_(world_size),
        watchdog_interval_ns_(watchdog_interval_ns) {}
  Status fail_once(EngineSupervisionFailure failure,
                   bool automatic_restart_eligible, Status cause);
  EngineGenerationLifecycle lifecycle_;
  EngineProgressWatchdog watchdog_;
  EngineAllocationLeaseExpectation lease_expectation_;
  std::uint32_t world_size_ = 0;
  std::uint64_t watchdog_interval_ns_ = 0;
  EngineSupervisionFailure first_failure_ = EngineSupervisionFailure::kNone;
};

}  // namespace pih
