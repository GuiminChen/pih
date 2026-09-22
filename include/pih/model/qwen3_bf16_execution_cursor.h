#pragma once

#include <cstddef>
#include <cstdint>

#include "pih/core/result.h"
#include "pih/core/status.h"
#include "pih/model/qwen3_bf16_execution_schedule.h"

namespace pih {

enum class QwenBf16ExecutionState : std::uint8_t {
  kReady,
  kRunning,
  kCompleted,
  kPoisoned,
};

struct QwenBf16ExecutionRequest final {
  std::uint64_t tokens;
  std::uint64_t first_position;
  std::uint64_t active_logit_rows;
  std::uint64_t generation;

  bool operator==(const QwenBf16ExecutionRequest&) const = default;
};

class QwenBf16ExecutionDriver {
 public:
  virtual ~QwenBf16ExecutionDriver() = default;
  virtual Status Execute(const QwenBf16ExecutionStep& step,
                         const QwenBf16ExecutionRequest& request) = 0;
};

class QwenBf16ExecutionCursor final {
 public:
  static Result<QwenBf16ExecutionCursor> Create(const Qwen3Config& config);

  Status Begin(std::uint64_t tokens, std::uint64_t first_position);
  Status RunNext(QwenBf16ExecutionDriver& driver);

  [[nodiscard]] QwenBf16ExecutionState state() const noexcept { return state_; }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return request_.generation;
  }
  [[nodiscard]] std::size_t next_step() const noexcept { return next_step_; }
  [[nodiscard]] const QwenBf16ExecutionRequest& request() const noexcept {
    return request_;
  }

 private:
  explicit QwenBf16ExecutionCursor(QwenBf16ExecutionSchedule schedule) noexcept
      : schedule_(schedule) {}

  QwenBf16ExecutionSchedule schedule_;
  QwenBf16ExecutionRequest request_{};
  std::size_t next_step_ = 0;
  QwenBf16ExecutionState state_ = QwenBf16ExecutionState::kReady;
};

}  // namespace pih
