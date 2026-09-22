#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

enum class EngineShutdownState : std::uint8_t {
  kRunning,
  kDraining,
  kStopping,
  kForceStopping,
  kComplete,
};

enum class EngineShutdownAction : std::uint8_t {
  kNone,
  kWithdrawReadiness,
  kBeginStop,
  kForceKillDomain,
};

struct EngineShutdownPolicy final {
  std::uint64_t graceful_drain_ns = 0;
  std::uint64_t force_stop_ns = 0;
};

class EngineShutdownController final {
 public:
  static Result<EngineShutdownController> Create(EngineShutdownPolicy policy);
  Result<EngineShutdownAction> request_termination(std::uint64_t now_ns);
  Result<EngineShutdownAction> abort_start(std::uint64_t now_ns);
  Result<EngineShutdownAction> poll(std::uint64_t now_ns,
                                    bool committed_work_remaining);
  Status mark_domain_empty();

  [[nodiscard]] EngineShutdownState state() const noexcept { return state_; }
  [[nodiscard]] std::uint64_t drain_deadline_ns() const noexcept {
    return drain_deadline_ns_;
  }
  [[nodiscard]] std::uint64_t force_deadline_ns() const noexcept {
    return force_deadline_ns_;
  }

 private:
  explicit EngineShutdownController(EngineShutdownPolicy policy) noexcept
      : policy_(policy) {}
  EngineShutdownPolicy policy_{};
  EngineShutdownState state_ = EngineShutdownState::kRunning;
  std::uint64_t drain_deadline_ns_ = 0;
  std::uint64_t force_deadline_ns_ = 0;
  std::uint64_t last_now_ns_ = 0;
  bool force_action_emitted_ = false;
};

}  // namespace pih
