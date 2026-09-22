#pragma once

#include <cstdint>

#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

struct EngineProgressHeartbeat final {
  std::uint64_t engine_generation = 0;
  std::uint64_t controller_progress = 0;
  std::vector<std::uint64_t> rank_progress;
  std::uint64_t observed_at_ns = 0;
};

class EngineProgressWatchdog final {
 public:
  static Result<EngineProgressWatchdog> Create(
      std::uint64_t engine_generation, std::uint32_t world_size,
      std::uint64_t deadline_interval_ns);
  Status arm(std::uint64_t now_ns);
  Status accept(const EngineProgressHeartbeat& heartbeat,
                std::uint64_t now_ns);
  Status poll(std::uint64_t now_ns);
  [[nodiscard]] bool expired() const noexcept { return expired_; }
  [[nodiscard]] std::uint64_t deadline_ns() const noexcept {
    return deadline_ns_;
  }

 private:
  EngineProgressWatchdog(std::uint64_t generation, std::uint32_t world_size,
                         std::uint64_t interval) noexcept
      : generation_(generation), world_size_(world_size), interval_(interval),
        rank_progress_(world_size, 0) {}
  Status expire() noexcept;
  std::uint64_t generation_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint64_t interval_ = 0;
  std::uint64_t controller_progress_ = 0;
  std::vector<std::uint64_t> rank_progress_;
  std::uint64_t deadline_ns_ = 0;
  std::uint64_t last_now_ns_ = 0;
  std::uint64_t last_observed_at_ns_ = 0;
  bool armed_ = false;
  bool expired_ = false;
};

}  // namespace pih
