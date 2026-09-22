#include "pih/model/engine_supervision_coordinator.h"

namespace pih {

Result<EngineSupervisionCoordinator> EngineSupervisionCoordinator::Create(
    std::uint64_t generation, std::uint32_t world_size,
    std::uint64_t watchdog_interval_ns, EngineRestartPolicy restart_policy,
    EngineAllocationLeaseExpectation lease_expectation) {
  EngineAllocationLeaseObservation self{
      lease_expectation.lease_digest, lease_expectation.filesystem_identity,
      lease_expectation.file_identity, true, true};
  const auto valid = verify_engine_allocation_lease(lease_expectation, self);
  if (!valid.ok()) return valid;
  auto lifecycle = EngineGenerationLifecycle::Create(
      generation, lease_expectation.lease_digest, restart_policy);
  if (!lifecycle.ok()) return lifecycle.status();
  auto watchdog = EngineProgressWatchdog::Create(
      generation, world_size, watchdog_interval_ns);
  if (!watchdog.ok()) return watchdog.status();
  return EngineSupervisionCoordinator(std::move(*lifecycle),
                                      std::move(*watchdog), lease_expectation,
                                      world_size, watchdog_interval_ns);
}

Status EngineSupervisionCoordinator::fail_once(
    EngineSupervisionFailure failure, bool automatic_restart_eligible,
    Status cause) {
  if (first_failure_ != EngineSupervisionFailure::kNone)
    return Status::FailedPrecondition(
        "engine supervision generation already failed");
  const auto failed =
      lifecycle_.fail_generation(automatic_restart_eligible);
  if (!failed.ok()) return failed;
  first_failure_ = failure;
  return cause.ok() ? Status::Internal("engine supervision failure") : cause;
}

Status EngineSupervisionCoordinator::begin_start(
    const EngineAllocationLeaseObservation& lease, std::uint64_t now_ns) {
  if (first_failure_ != EngineSupervisionFailure::kNone)
    return Status::FailedPrecondition("engine supervision generation failed");
  auto status = lifecycle_.begin_start();
  if (!status.ok()) return status;
  status = verify_engine_allocation_lease(lease_expectation_, lease);
  if (!status.ok())
    return fail_once(EngineSupervisionFailure::kAllocationLeaseDrift, false,
                     status);
  status = watchdog_.arm(now_ns);
  if (!status.ok())
    return fail_once(EngineSupervisionFailure::kProgressWatchdogExpired, true,
                     status);
  return Status::Ok();
}

Status EngineSupervisionCoordinator::publish_ready(
    const EngineAllocationLeaseObservation& lease) {
  if (first_failure_ != EngineSupervisionFailure::kNone)
    return Status::FailedPrecondition("engine supervision generation failed");
  const auto status = verify_engine_allocation_lease(lease_expectation_, lease);
  if (!status.ok())
    return fail_once(EngineSupervisionFailure::kAllocationLeaseDrift, false,
                     status);
  return lifecycle_.publish_ready();
}

Status EngineSupervisionCoordinator::accept_heartbeat(
    const EngineProgressHeartbeat& heartbeat, std::uint64_t now_ns) {
  if (first_failure_ != EngineSupervisionFailure::kNone)
    return Status::FailedPrecondition("engine supervision generation failed");
  const auto status = watchdog_.accept(heartbeat, now_ns);
  if (!status.ok())
    return fail_once(EngineSupervisionFailure::kProgressWatchdogExpired, true,
                     status);
  return Status::Ok();
}

Status EngineSupervisionCoordinator::poll(
    const EngineAllocationLeaseObservation& lease, bool all_ranks_healthy,
    std::uint64_t now_ns) {
  if (first_failure_ != EngineSupervisionFailure::kNone)
    return Status::FailedPrecondition("engine supervision generation failed");
  auto status = verify_engine_allocation_lease(lease_expectation_, lease);
  if (!status.ok())
    return fail_once(EngineSupervisionFailure::kAllocationLeaseDrift, false,
                     status);
  if (!all_ranks_healthy)
    return fail_once(EngineSupervisionFailure::kRankLoss, true,
                     Status::Unavailable("engine rank was lost"));
  status = watchdog_.poll(now_ns);
  if (!status.ok())
    return fail_once(EngineSupervisionFailure::kProgressWatchdogExpired, true,
                     status);
  return Status::Ok();
}

Status EngineSupervisionCoordinator::report_rank_loss() {
  return fail_once(EngineSupervisionFailure::kRankLoss, true,
                   Status::Unavailable("engine rank was lost"));
}

Result<EngineShutdownAction>
EngineSupervisionCoordinator::request_termination(
    EngineShutdownController& shutdown, std::uint64_t now_ns) {
  if (first_failure_ != EngineSupervisionFailure::kNone)
    return Status::FailedPrecondition(
        "failed engine generation cannot begin planned shutdown");
  const bool first_request =
      lifecycle_.state() == EngineGenerationState::kReady &&
      shutdown.state() == EngineShutdownState::kRunning;
  const bool repeated_request =
      (lifecycle_.state() == EngineGenerationState::kDraining ||
       lifecycle_.state() == EngineGenerationState::kStopping) &&
      (shutdown.state() == EngineShutdownState::kDraining ||
       shutdown.state() == EngineShutdownState::kStopping ||
       shutdown.state() == EngineShutdownState::kForceStopping);
  if (!first_request && !repeated_request)
    return Status::FailedPrecondition(
        "planned shutdown state is not correlated with engine lifecycle");
  auto action = shutdown.request_termination(now_ns);
  if (!action.ok()) return action.status();
  if (*action == EngineShutdownAction::kWithdrawReadiness) {
    const auto status = lifecycle_.begin_drain();
    if (!status.ok()) return status;
  } else if (*action == EngineShutdownAction::kForceKillDomain &&
             lifecycle_.state() == EngineGenerationState::kDraining) {
    const auto status = lifecycle_.begin_stop();
    if (!status.ok()) return status;
  }
  return *action;
}

Result<EngineShutdownAction>
EngineSupervisionCoordinator::consume_termination_event(
    EngineTerminationEventGate& gate, const EngineTerminationEvent& event,
    EngineShutdownController& shutdown, std::uint64_t now_ns) {
  auto accepted = gate.accept(event);
  if (!accepted.ok()) return accepted.status();
  if (lifecycle_.state() == EngineGenerationState::kStarting) {
    auto action = shutdown.abort_start(now_ns);
    if (!action.ok()) return action.status();
    const auto status = lifecycle_.abort_start();
    if (!status.ok()) return status;
    return *action;
  }
  return request_termination(shutdown, now_ns);
}

Result<EngineShutdownAction> EngineSupervisionCoordinator::poll_shutdown(
    EngineShutdownController& shutdown, std::uint64_t now_ns,
    bool committed_work_remaining) {
  if (first_failure_ != EngineSupervisionFailure::kNone)
    return Status::FailedPrecondition(
        "failed engine generation cannot poll planned shutdown");
  if ((lifecycle_.state() != EngineGenerationState::kDraining &&
       lifecycle_.state() != EngineGenerationState::kStopping) ||
      (shutdown.state() != EngineShutdownState::kDraining &&
       shutdown.state() != EngineShutdownState::kStopping &&
       shutdown.state() != EngineShutdownState::kForceStopping))
    return Status::FailedPrecondition(
        "planned shutdown state is not correlated with engine lifecycle");
  auto action = shutdown.poll(now_ns, committed_work_remaining);
  if (!action.ok()) return action.status();
  if ((*action == EngineShutdownAction::kBeginStop ||
       *action == EngineShutdownAction::kForceKillDomain) &&
      lifecycle_.state() == EngineGenerationState::kDraining) {
    const auto status = lifecycle_.begin_stop();
    if (!status.ok()) return status;
  }
  return *action;
}

Status EngineSupervisionCoordinator::mark_shutdown_domain_empty(
    EngineShutdownController& shutdown) {
  if (first_failure_ != EngineSupervisionFailure::kNone)
    return Status::FailedPrecondition(
        "failed engine generation is not a planned shutdown");
  if (lifecycle_.state() != EngineGenerationState::kStopping ||
      (shutdown.state() != EngineShutdownState::kStopping &&
       shutdown.state() != EngineShutdownState::kForceStopping))
    return Status::FailedPrecondition(
        "planned shutdown is not ready for domain completion");
  auto status = lifecycle_.controller_reaped();
  if (!status.ok()) return status;
  return shutdown.mark_domain_empty();
}

Status EngineSupervisionCoordinator::execute_force_kill(
    EngineShutdownAction action, const EngineShutdownController& shutdown,
    EngineGenerationDomainTermination& termination) {
  if (first_failure_ != EngineSupervisionFailure::kNone ||
      lifecycle_.state() != EngineGenerationState::kStopping ||
      shutdown.state() != EngineShutdownState::kForceStopping ||
      action != EngineShutdownAction::kForceKillDomain)
    return Status::FailedPrecondition(
        "engine generation is not authorized for planned force kill");
  return termination.force_kill();
}

Result<bool> EngineSupervisionCoordinator::reconcile_shutdown_domain(
    EngineShutdownController& shutdown,
    EngineGenerationDomainTermination& termination) {
  if (first_failure_ != EngineSupervisionFailure::kNone ||
      lifecycle_.state() != EngineGenerationState::kStopping)
    return Status::FailedPrecondition(
        "engine generation is not reconciling planned shutdown");
  auto empty = termination.poll_empty();
  if (!empty.ok()) return empty.status();
  if (!*empty) return false;
  const auto status = mark_shutdown_domain_empty(shutdown);
  if (!status.ok()) return status;
  return true;
}

Status EngineSupervisionCoordinator::controller_reaped() {
  if (first_failure_ == EngineSupervisionFailure::kNone)
    return Status::FailedPrecondition(
        "engine supervision has no failed generation to reap");
  return lifecycle_.controller_reaped();
}

Status EngineSupervisionCoordinator::begin_quiescence_check() {
  return lifecycle_.begin_quiescence_check();
}

Result<EngineQuiescenceState>
EngineSupervisionCoordinator::observe_quiescence(
    const EngineGenerationQuiescenceReceipt& receipt,
    const EnginePidfdReapingLedger& pidfd_ledger,
    const EngineGpuProcessInventoryGate& gpu_inventory,
    const EngineOwnedResourceBaselineGate& owned_resources,
    std::uint64_t now_ns) {
  auto authoritative = receipt;
  authoritative.observation.pidfds_reaped = pidfd_ledger.complete();
  const bool inventory_bound = gpu_inventory.bound_to(
      lifecycle_.generation(), lease_expectation_.lease_digest);
  authoritative.observation.target_gpu_processes_absent =
      inventory_bound && !gpu_inventory.poisoned() &&
      gpu_inventory.state() == EngineGpuProcessInventoryState::kEmpty;
  authoritative.observation.shm_baseline =
      owned_resources.at_baseline(EngineOwnedResourceKind::kShm);
  authoritative.observation.network_baseline =
      owned_resources.at_baseline(EngineOwnedResourceKind::kNetwork);
  authoritative.observation.pinned_memory_baseline =
      owned_resources.at_baseline(EngineOwnedResourceKind::kPinnedMemory);
  authoritative.observation.listener_baseline =
      owned_resources.at_baseline(EngineOwnedResourceKind::kListener);
  authoritative.observation.artifact_baseline =
      owned_resources.at_baseline(EngineOwnedResourceKind::kArtifact);
  authoritative.observation.gpu_allocation_baseline =
      owned_resources.at_baseline(EngineOwnedResourceKind::kGpuAllocation);
  if (pidfd_ledger.poisoned())
    authoritative.observation.visibility_complete = false;
  if (!inventory_bound || gpu_inventory.poisoned() ||
      gpu_inventory.state() == EngineGpuProcessInventoryState::kUnknown)
    authoritative.observation.visibility_complete = false;
  const bool resources_bound = owned_resources.bound_to(
      lifecycle_.generation(), lease_expectation_.lease_digest);
  if (!resources_bound || owned_resources.poisoned() ||
      owned_resources.state() == EngineOwnedResourceBaselineState::kUnknown)
    authoritative.observation.visibility_complete = false;
  return lifecycle_.observe_quiescence(authoritative, now_ns);
}

Status EngineSupervisionCoordinator::authorize_restart(
    std::uint64_t now_ns, std::uint64_t next_generation,
    const EngineAllocationLeaseObservation& lease) {
  if (first_failure_ == EngineSupervisionFailure::kNone) {
    return Status::FailedPrecondition(
        "engine supervision has no failed generation to restart");
  }
  const auto lease_status =
      verify_engine_allocation_lease(lease_expectation_, lease);
  if (!lease_status.ok()) {
    (void)lifecycle_.quarantine();
    return lease_status;
  }
  auto next_watchdog = EngineProgressWatchdog::Create(
      next_generation, world_size_, watchdog_interval_ns_);
  if (!next_watchdog.ok()) return next_watchdog.status();
  const auto rollover =
      lifecycle_.finish_backoff(now_ns, next_generation);
  if (!rollover.ok()) return rollover;
  watchdog_ = std::move(*next_watchdog);
  first_failure_ = EngineSupervisionFailure::kNone;
  return Status::Ok();
}

bool EngineSupervisionCoordinator::ready() const noexcept {
  return first_failure_ == EngineSupervisionFailure::kNone &&
         lifecycle_.state() == EngineGenerationState::kReady;
}

}  // namespace pih
