#include "pih/model/engine_progress_watchdog.h"

#include <limits>

namespace pih {

Result<EngineProgressWatchdog> EngineProgressWatchdog::Create(
    std::uint64_t engine_generation, std::uint32_t world_size,
    std::uint64_t deadline_interval_ns) {
  if (engine_generation == 0 || world_size == 0 || world_size > 4 ||
      deadline_interval_ns == 0)
    return Status::InvalidArgument("engine progress watchdog is invalid");
  return EngineProgressWatchdog(engine_generation, world_size,
                                deadline_interval_ns);
}

Status EngineProgressWatchdog::expire() noexcept {
  expired_ = true;
  return Status::DeadlineExceeded("engine progress watchdog expired");
}

Status EngineProgressWatchdog::arm(std::uint64_t now_ns) {
  if (armed_ || expired_)
    return Status::FailedPrecondition("engine progress watchdog is closed");
  if (now_ns > std::numeric_limits<std::uint64_t>::max() - interval_)
    return expire();
  armed_ = true;
  last_now_ns_ = now_ns;
  last_observed_at_ns_ = now_ns;
  deadline_ns_ = now_ns + interval_;
  return Status::Ok();
}

Status EngineProgressWatchdog::accept(
    const EngineProgressHeartbeat& heartbeat, std::uint64_t now_ns) {
  if (!armed_ || expired_)
    return Status::FailedPrecondition("engine progress watchdog is closed");
  if (now_ns < last_now_ns_ || now_ns >= deadline_ns_) return expire();
  if (heartbeat.engine_generation != generation_ ||
      heartbeat.controller_progress == 0 ||
      heartbeat.controller_progress <= controller_progress_ ||
      heartbeat.rank_progress.size() != world_size_ ||
      heartbeat.observed_at_ns <= last_observed_at_ns_ ||
      heartbeat.observed_at_ns > now_ns) {
    return expire();
  }
  for (std::size_t rank = 0; rank < rank_progress_.size(); ++rank) {
    if (heartbeat.rank_progress[rank] == 0 ||
        heartbeat.rank_progress[rank] <= rank_progress_[rank])
      return expire();
  }
  if (heartbeat.observed_at_ns >
      std::numeric_limits<std::uint64_t>::max() - interval_)
    return expire();
  controller_progress_ = heartbeat.controller_progress;
  rank_progress_ = heartbeat.rank_progress;
  last_observed_at_ns_ = heartbeat.observed_at_ns;
  last_now_ns_ = now_ns;
  deadline_ns_ = heartbeat.observed_at_ns + interval_;
  if (now_ns >= deadline_ns_) return expire();
  return Status::Ok();
}

Status EngineProgressWatchdog::poll(std::uint64_t now_ns) {
  if (!armed_ || expired_)
    return Status::FailedPrecondition("engine progress watchdog is closed");
  if (now_ns < last_now_ns_ || now_ns >= deadline_ns_) return expire();
  last_now_ns_ = now_ns;
  return Status::Ok();
}

}  // namespace pih
