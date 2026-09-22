#pragma once

#include <atomic>
#include <cstdint>

#include "pih/core/status.h"

namespace pih {

class DeepSeekBoundaryHealthState final {
 public:
  Status report_device_error(std::uint32_t error_code) noexcept;
  void poison() noexcept {
    engine_poisoned_.store(true, std::memory_order_release);
  }

  [[nodiscard]] std::uint32_t device_error_code() const noexcept {
    return device_error_code_.load(std::memory_order_acquire);
  }
  [[nodiscard]] bool engine_poisoned() const noexcept {
    return engine_poisoned_.load(std::memory_order_acquire);
  }
  [[nodiscard]] const std::atomic<std::uint32_t>& device_error_atomic()
      const noexcept { return device_error_code_; }
  [[nodiscard]] const std::atomic<bool>& engine_poisoned_atomic()
      const noexcept { return engine_poisoned_; }

 private:
  std::atomic<std::uint32_t> device_error_code_{0};
  std::atomic<bool> engine_poisoned_{false};
};

}  // namespace pih
