#include "pih/platform/linux/linux_deepseek_rank_worker_bootstrap_runtime.h"

#include <cerrno>
#include <climits>
#include <limits>
#include <string>

#include <poll.h>
#include <time.h>

namespace pih {
namespace {

Status failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

}  // namespace

Result<std::uint64_t>
LinuxDeepSeekRankWorkerBootstrapRuntime::monotonic_now_ns() {
  timespec value{};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0)
    return failure("clock_gettime CLOCK_MONOTONIC");
  if (value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000L)
    return Status::Internal("CLOCK_MONOTONIC returned an invalid timestamp");
  constexpr auto billion = std::uint64_t{1000000000};
  const auto seconds = static_cast<std::uint64_t>(value.tv_sec);
  if (seconds > std::numeric_limits<std::uint64_t>::max() / billion)
    return Status::Internal("CLOCK_MONOTONIC timestamp overflowed");
  return seconds * billion + static_cast<std::uint64_t>(value.tv_nsec);
}

Status LinuxDeepSeekRankWorkerBootstrapRuntime::wait_for_control(
    std::int32_t control_fd, DeepSeekRankWorkerWaitEvent event,
    std::uint64_t startup_deadline_ns) {
  if (control_fd < 0 || startup_deadline_ns == 0)
    return Status::InvalidArgument("DeepSeek worker control wait is invalid");
  for (;;) {
    auto now = monotonic_now_ns();
    if (!now.ok()) return now.status();
    if (*now >= startup_deadline_ns) return Status::Ok();
    const auto remaining = startup_deadline_ns - *now;
    const auto remaining_ms = remaining / 1000000ULL +
                              (remaining % 1000000ULL != 0 ? 1ULL : 0ULL);
    const auto timeout_ms = static_cast<int>(
        remaining_ms > static_cast<std::uint64_t>(INT_MAX)
            ? INT_MAX
            : remaining_ms);
    pollfd descriptor{control_fd,
                      static_cast<short>(
                          event == DeepSeekRankWorkerWaitEvent::kChallengeReadable
                              ? POLLIN
                              : POLLOUT),
                      0};
    const auto observed = ::poll(&descriptor, 1, timeout_ms);
    if (observed < 0 && errno == EINTR) continue;
    if (observed < 0) return failure("poll DeepSeek worker control socket");
    if (observed == 0) return Status::Ok();
    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
      return Status::FailedPrecondition(
          "DeepSeek worker control socket closed during startup");
    if ((descriptor.revents & descriptor.events) != 0) return Status::Ok();
  }
}

}  // namespace pih
