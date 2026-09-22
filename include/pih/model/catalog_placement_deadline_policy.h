#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "pih/core/result.h"

namespace pih {

// Immutable, controller-clock-relative budgets for placement_deadline_policy_v1.
// BEGIN turns these into absolute timestamps in its one monotonic clock domain.
class CatalogPlacementDeadlinePolicy final {
 public:
  static constexpr std::uint64_t kMaximumTotalMs = 3'600'000;

  static Result<CatalogPlacementDeadlinePolicy> Create(
      std::uint64_t verify_ms, std::uint64_t startup_ms,
      std::uint64_t route_ms, std::uint64_t cleanup_ms,
      std::uint64_t total_ms) {
    if (verify_ms == 0 || startup_ms == 0 || route_ms == 0 ||
        cleanup_ms == 0 || total_ms == 0 || total_ms > kMaximumTotalMs ||
        verify_ms > startup_ms || startup_ms > route_ms ||
        route_ms > total_ms || cleanup_ms > total_ms) {
      return Status::InvalidArgument("placement deadline policy is invalid");
    }
    return CatalogPlacementDeadlinePolicy(verify_ms, startup_ms, route_ms,
                                          cleanup_ms, total_ms);
  }

  Result<std::array<std::uint64_t, 5>> absolute_deadlines(
      std::uint64_t begin_monotonic_ms) const {
    if (begin_monotonic_ms == 0 ||
        begin_monotonic_ms > std::numeric_limits<std::uint64_t>::max() - total_ms_) {
      return Status::InvalidArgument("placement monotonic clock is invalid");
    }
    return std::array<std::uint64_t, 5>{begin_monotonic_ms + verify_ms_,
                                        begin_monotonic_ms + startup_ms_,
                                        begin_monotonic_ms + route_ms_,
                                        begin_monotonic_ms + cleanup_ms_,
                                        begin_monotonic_ms + total_ms_};
  }

 private:
  CatalogPlacementDeadlinePolicy(std::uint64_t verify_ms,
                                 std::uint64_t startup_ms,
                                 std::uint64_t route_ms,
                                 std::uint64_t cleanup_ms,
                                 std::uint64_t total_ms) noexcept
      : verify_ms_(verify_ms), startup_ms_(startup_ms), route_ms_(route_ms),
        cleanup_ms_(cleanup_ms), total_ms_(total_ms) {}
  std::uint64_t verify_ms_;
  std::uint64_t startup_ms_;
  std::uint64_t route_ms_;
  std::uint64_t cleanup_ms_;
  std::uint64_t total_ms_;
};

}  // namespace pih
