#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>

#include "pih/core/result.h"

namespace pih {

enum class QwenEngineCloseState : std::uint8_t {
  kOpen,
  kClosing,
  kClosed,
};

class QwenEngineCloseGate;

class QwenEngineOperationLease final {
 public:
  QwenEngineOperationLease(const QwenEngineOperationLease&) = delete;
  QwenEngineOperationLease& operator=(const QwenEngineOperationLease&) = delete;
  QwenEngineOperationLease(QwenEngineOperationLease&& other) noexcept;
  QwenEngineOperationLease& operator=(QwenEngineOperationLease&&) = delete;
  ~QwenEngineOperationLease();

 private:
  friend class QwenEngineCloseGate;
  explicit QwenEngineOperationLease(QwenEngineCloseGate& gate) noexcept
      : gate_(&gate) {}
  QwenEngineCloseGate* gate_;
};

class QwenEngineCloseGate final {
 public:
  Status require_open() const;
  Result<QwenEngineOperationLease> enter();
  bool request_close();
  void wait_drained();
  void finish_close(Status outcome) noexcept;
  Status wait_closed();

  [[nodiscard]] QwenEngineCloseState state() const noexcept {
    return state_.load(std::memory_order_acquire);
  }

 private:
  friend class QwenEngineOperationLease;
  void release() noexcept;

  std::atomic<QwenEngineCloseState> state_{QwenEngineCloseState::kOpen};
  std::mutex wait_mutex_;
  std::condition_variable wait_condition_;
  std::uint64_t active_operations_ = 0;
  std::optional<Status> close_outcome_;
};

}  // namespace pih
