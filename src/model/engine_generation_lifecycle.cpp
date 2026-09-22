#include "pih/model/engine_generation_lifecycle.h"

#include <algorithm>
#include <limits>

namespace pih {
namespace {

bool all_resources_at_baseline(
    const EngineGenerationQuiescenceObservation& value) {
  return value.cgroup_domain_empty && value.pidfds_reaped &&
         value.target_gpu_processes_absent && value.shm_baseline &&
         value.network_baseline && value.pinned_memory_baseline &&
         value.listener_baseline && value.artifact_baseline &&
         value.gpu_allocation_baseline;
}

}  // namespace

Result<EngineGenerationLifecycle> EngineGenerationLifecycle::Create(
    std::uint64_t generation, Sha256Digest allocation_lease_digest,
    EngineRestartPolicy policy) {
  if (generation == 0 || allocation_lease_digest == Sha256Digest{} ||
      policy.backoff_initial_ns == 0 ||
      policy.backoff_max_ns == 0 ||
      policy.backoff_initial_ns > policy.backoff_max_ns ||
      policy.restart_window_ns == 0 || policy.restart_burst_max == 0) {
    return Status::InvalidArgument(
        "engine generation restart policy is invalid");
  }
  return EngineGenerationLifecycle(generation, allocation_lease_digest, policy);
}

Status EngineGenerationLifecycle::require(EngineGenerationState expected,
                                          EngineGenerationState next) {
  if (state_ != expected)
    return Status::FailedPrecondition(
        "engine generation lifecycle transition is invalid");
  state_ = next;
  return Status::Ok();
}

Status EngineGenerationLifecycle::begin_start() {
  return require(EngineGenerationState::kLeased,
                 EngineGenerationState::kStarting);
}

Status EngineGenerationLifecycle::publish_ready() {
  return require(EngineGenerationState::kStarting,
                 EngineGenerationState::kReady);
}

Status EngineGenerationLifecycle::begin_drain() {
  return require(EngineGenerationState::kReady,
                 EngineGenerationState::kDraining);
}

Status EngineGenerationLifecycle::abort_start() {
  const auto status = require(EngineGenerationState::kStarting,
                              EngineGenerationState::kStopping);
  if (status.ok()) automatic_restart_eligible_ = false;
  return status;
}

Status EngineGenerationLifecycle::begin_stop() {
  if (state_ != EngineGenerationState::kDraining &&
      state_ != EngineGenerationState::kFailed)
    return Status::FailedPrecondition(
        "engine generation cannot begin stopping");
  state_ = EngineGenerationState::kStopping;
  automatic_restart_eligible_ = false;
  return Status::Ok();
}

Status EngineGenerationLifecycle::fail_generation(
    bool automatic_restart_eligible) {
  if (state_ != EngineGenerationState::kStarting &&
      state_ != EngineGenerationState::kReady &&
      state_ != EngineGenerationState::kDraining &&
      state_ != EngineGenerationState::kStopping)
    return Status::FailedPrecondition("engine generation cannot fail");
  state_ = EngineGenerationState::kFailed;
  automatic_restart_eligible_ = automatic_restart_eligible;
  return Status::Ok();
}

Status EngineGenerationLifecycle::controller_reaped() {
  if (state_ != EngineGenerationState::kFailed &&
      state_ != EngineGenerationState::kStopping)
    return Status::FailedPrecondition(
        "engine generation controller is not reapable");
  state_ = EngineGenerationState::kDead;
  return Status::Ok();
}

Status EngineGenerationLifecycle::begin_quiescence_check() {
  return require(EngineGenerationState::kDead,
                 EngineGenerationState::kQuiescenceCheck);
}

Result<EngineQuiescenceState> EngineGenerationLifecycle::observe_quiescence(
    const EngineGenerationQuiescenceReceipt& receipt,
    std::uint64_t now_ns) {
  if (state_ != EngineGenerationState::kQuiescenceCheck)
    return Status::FailedPrecondition(
        "engine generation is not checking quiescence");
  if (receipt.generation != generation_ ||
      receipt.allocation_lease_digest != allocation_lease_digest_ ||
      receipt.sample_identity == 0 ||
      receipt.sample_identity <= last_quiescence_sample_identity_ ||
      receipt.sample_started_ns == 0 ||
      receipt.sample_completed_ns < receipt.sample_started_ns ||
      receipt.sample_completed_ns > now_ns ||
      receipt.sample_started_ns < last_quiescence_sample_completed_ns_) {
    state_ = EngineGenerationState::kQuarantined;
    return Status::FailedPrecondition(
        "engine generation quiescence receipt identity is invalid");
  }
  last_quiescence_sample_identity_ = receipt.sample_identity;
  last_quiescence_sample_completed_ns_ = receipt.sample_completed_ns;
  const auto& observation = receipt.observation;
  if (!observation.visibility_complete) {
    state_ = EngineGenerationState::kQuarantined;
    return EngineQuiescenceState::kUnknown;
  }
  if (!all_resources_at_baseline(observation))
    return EngineQuiescenceState::kPending;

  if (!automatic_restart_eligible_) {
    state_ = EngineGenerationState::kDead;
    return EngineQuiescenceState::kVerified;
  }
  if (last_restart_observation_ns_ != 0 &&
      now_ns < last_restart_observation_ns_) {
    state_ = EngineGenerationState::kQuarantined;
    return Status::FailedPrecondition(
        "engine generation restart clock regressed");
  }
  last_restart_observation_ns_ = now_ns;

  while (!attempts_.empty() && now_ns >= attempts_.front() &&
         now_ns - attempts_.front() >= policy_.restart_window_ns)
    attempts_.pop_front();
  if (attempts_.size() >= policy_.restart_burst_max) {
    state_ = EngineGenerationState::kQuarantined;
    return EngineQuiescenceState::kVerified;
  }
  attempts_.push_back(now_ns);
  std::uint64_t backoff = policy_.backoff_initial_ns;
  for (std::size_t index = 1; index < attempts_.size(); ++index) {
    if (backoff >= policy_.backoff_max_ns - backoff) {
      backoff = policy_.backoff_max_ns;
      break;
    }
    backoff = std::min(backoff * 2, policy_.backoff_max_ns);
  }
  if (now_ns > std::numeric_limits<std::uint64_t>::max() - backoff) {
    state_ = EngineGenerationState::kQuarantined;
    return Status::ResourceExhausted(
        "engine generation retry deadline overflowed");
  }
  retry_not_before_ns_ = now_ns + backoff;
  state_ = EngineGenerationState::kBackoff;
  return EngineQuiescenceState::kVerified;
}

Status EngineGenerationLifecycle::finish_backoff(
    std::uint64_t now_ns, std::uint64_t next_generation) {
  if (state_ != EngineGenerationState::kBackoff ||
      now_ns < retry_not_before_ns_ ||
      generation_ == std::numeric_limits<std::uint64_t>::max() ||
      next_generation != generation_ + 1)
    return Status::FailedPrecondition(
        "engine generation backoff is not complete");
  generation_ = next_generation;
  retry_not_before_ns_ = 0;
  automatic_restart_eligible_ = false;
  state_ = EngineGenerationState::kLeased;
  return Status::Ok();
}

Status EngineGenerationLifecycle::quarantine() noexcept {
  if (state_ != EngineGenerationState::kBackoff) {
    return Status::FailedPrecondition(
        "engine generation cannot enter restart quarantine");
  }
  retry_not_before_ns_ = 0;
  state_ = EngineGenerationState::kQuarantined;
  return Status::Ok();
}

}  // namespace pih
