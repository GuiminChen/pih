#include "pih/model/engine_supervisor_shutdown_handler.h"

namespace pih {

Result<EngineSupervisorShutdownHandler>
EngineSupervisorShutdownHandler::Create(std::uint64_t engine_generation,
                                        std::uint64_t engine_epoch) {
  auto gate = EngineSupervisorShutdownRequestGate::Create(engine_generation,
                                                          engine_epoch);
  if (!gate.ok()) return gate.status();
  return EngineSupervisorShutdownHandler(engine_generation, engine_epoch,
                                         std::move(*gate));
}

Result<EngineSupervisorShutdownHandlingResult>
EngineSupervisorShutdownHandler::accept(
    const EngineSupervisorShutdownRequest& request, std::uint64_t now_ns,
    EngineSupervisionCoordinator& coordinator,
    EngineShutdownController& shutdown,
    EngineGenerationDomainTermination& termination) {
  const bool can_begin =
      coordinator.ready() && coordinator.generation() == engine_generation_ &&
      shutdown.state() == EngineShutdownState::kRunning;
  const bool can_retry_kill =
      coordinator.generation() == engine_generation_ &&
      coordinator.lifecycle_state() == EngineGenerationState::kStopping &&
      shutdown.state() == EngineShutdownState::kForceStopping &&
      !termination.force_kill_issued();
  if (!gate_.accepted() && !can_begin && !can_retry_kill) {
    return Status::FailedPrecondition(
        "engine generation is not ready for supervisor force stop");
  }

  auto candidate_gate = gate_;
  auto disposition = candidate_gate.accept(request, now_ns);
  if (!disposition.ok()) {
    gate_ = std::move(candidate_gate);
    return disposition.status();
  }

  EngineShutdownAction action = EngineShutdownAction::kNone;
  if (*disposition == EngineSupervisorShutdownAckDisposition::kAccepted) {
    if (can_begin) {
      auto withdraw = coordinator.request_termination(shutdown, now_ns);
      if (!withdraw.ok()) return withdraw.status();
      if (*withdraw != EngineShutdownAction::kWithdrawReadiness) {
        return Status::Internal(
            "supervisor force stop did not withdraw engine readiness");
      }
      auto force = coordinator.request_termination(shutdown, now_ns);
      if (!force.ok()) return force.status();
      if (*force != EngineShutdownAction::kForceKillDomain) {
        return Status::Internal(
            "supervisor force stop did not authorize domain kill");
      }
      action = *force;
    } else {
      action = EngineShutdownAction::kForceKillDomain;
    }
    const auto killed =
        coordinator.execute_force_kill(action, shutdown, termination);
    if (!killed.ok()) return killed;
  }

  gate_ = std::move(candidate_gate);

  return EngineSupervisorShutdownHandlingResult{
      {engine_generation_, engine_epoch_, request.request_identity, now_ns,
       *disposition},
      action};
}

}  // namespace pih
