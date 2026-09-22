#pragma once

#include <atomic>
#include <cstdint>

#include "pih/core/status.h"

namespace pih {

enum class QwenBf16RequestGateState : std::uint8_t {
  kReady,
  kRunning,
  kPoisoned,
  kClosed,
};

class QwenBf16RequestGate final {
 public:
  Status begin();
  Status finish(bool succeeded);
  Status close();

  [[nodiscard]] QwenBf16RequestGateState state() const noexcept {
    return state_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<QwenBf16RequestGateState> state_{
      QwenBf16RequestGateState::kReady};
};

}  // namespace pih
