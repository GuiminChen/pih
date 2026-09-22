#include "pih/model/engine_shutdown_controller.h"

#include <limits>

namespace pih {

Result<EngineShutdownController> EngineShutdownController::Create(
    EngineShutdownPolicy policy) {
  if (policy.graceful_drain_ns == 0 || policy.force_stop_ns == 0 ||
      policy.graceful_drain_ns >
          std::numeric_limits<std::uint64_t>::max() - policy.force_stop_ns)
    return Status::InvalidArgument("engine shutdown policy is invalid");
  return EngineShutdownController(policy);
}

Result<EngineShutdownAction> EngineShutdownController::request_termination(
    std::uint64_t now_ns) {
  if (state_ == EngineShutdownState::kComplete ||
      state_ == EngineShutdownState::kForceStopping)
    return EngineShutdownAction::kNone;
  if (state_ == EngineShutdownState::kRunning) {
    const auto total = policy_.graceful_drain_ns + policy_.force_stop_ns;
    if (now_ns > std::numeric_limits<std::uint64_t>::max() - total)
      return Status::ResourceExhausted(
          "engine shutdown deadline overflowed");
    last_now_ns_ = now_ns;
    drain_deadline_ns_ = now_ns + policy_.graceful_drain_ns;
    force_deadline_ns_ = now_ns + total;
    state_ = EngineShutdownState::kDraining;
    return EngineShutdownAction::kWithdrawReadiness;
  }
  if (now_ns < last_now_ns_)
    return Status::FailedPrecondition("engine shutdown clock regressed");
  last_now_ns_ = now_ns;
  state_ = EngineShutdownState::kForceStopping;
  if (force_action_emitted_) return EngineShutdownAction::kNone;
  force_action_emitted_ = true;
  return EngineShutdownAction::kForceKillDomain;
}

Result<EngineShutdownAction> EngineShutdownController::abort_start(
    std::uint64_t now_ns) {
  if (state_ != EngineShutdownState::kRunning)
    return Status::FailedPrecondition(
        "engine startup shutdown is already active");
  if (now_ns > std::numeric_limits<std::uint64_t>::max() -
                   policy_.force_stop_ns)
    return Status::ResourceExhausted(
        "engine startup shutdown deadline overflowed");
  last_now_ns_ = now_ns;
  drain_deadline_ns_ = now_ns;
  force_deadline_ns_ = now_ns + policy_.force_stop_ns;
  state_ = EngineShutdownState::kStopping;
  return EngineShutdownAction::kBeginStop;
}

Result<EngineShutdownAction> EngineShutdownController::poll(
    std::uint64_t now_ns, bool committed_work_remaining) {
  if (state_ == EngineShutdownState::kRunning ||
      state_ == EngineShutdownState::kComplete)
    return Status::FailedPrecondition("engine shutdown is not active");
  if (now_ns < last_now_ns_)
    return Status::FailedPrecondition("engine shutdown clock regressed");
  last_now_ns_ = now_ns;
  if (state_ == EngineShutdownState::kForceStopping)
    return EngineShutdownAction::kNone;
  if (now_ns >= force_deadline_ns_) {
    state_ = EngineShutdownState::kForceStopping;
    if (!force_action_emitted_) {
      force_action_emitted_ = true;
      return EngineShutdownAction::kForceKillDomain;
    }
    return EngineShutdownAction::kNone;
  }
  if (state_ == EngineShutdownState::kDraining &&
      (!committed_work_remaining || now_ns >= drain_deadline_ns_)) {
    state_ = EngineShutdownState::kStopping;
    return EngineShutdownAction::kBeginStop;
  }
  return EngineShutdownAction::kNone;
}

Status EngineShutdownController::mark_domain_empty() {
  if (state_ != EngineShutdownState::kStopping &&
      state_ != EngineShutdownState::kForceStopping)
    return Status::FailedPrecondition(
        "engine shutdown domain is not stopping");
  state_ = EngineShutdownState::kComplete;
  return Status::Ok();
}

}  // namespace pih
