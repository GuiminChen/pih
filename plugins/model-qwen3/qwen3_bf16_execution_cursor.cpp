#include "pih/model/qwen3_bf16_execution_cursor.h"

#include <limits>

#include "pih/model/qwen3_bf16_linear_shape.h"

namespace pih {

Result<QwenBf16ExecutionCursor> QwenBf16ExecutionCursor::Create(
    const Qwen3Config& config) {
  auto schedule = QwenBf16ExecutionSchedule::Create(config);
  if (!schedule.ok()) {
    return schedule.status();
  }
  return QwenBf16ExecutionCursor(std::move(*schedule));
}

Status QwenBf16ExecutionCursor::Begin(std::uint64_t tokens,
                                     std::uint64_t first_position) {
  if (state_ == QwenBf16ExecutionState::kRunning) {
    return Status::FailedPrecondition(
        "Qwen BF16 execution request is already running");
  }
  if (state_ == QwenBf16ExecutionState::kPoisoned) {
    return Status::FailedPrecondition(
        "Qwen BF16 execution cursor is permanently poisoned");
  }
  if (tokens == 0 ||
      tokens > QwenBf16LinearShape::kMaximumTokensPerPlan) {
    return Status::InvalidArgument(
        "Qwen BF16 execution token count is outside the frozen plan bound");
  }
  constexpr std::uint64_t kMaximumPositions = 40960;
  if (first_position >= kMaximumPositions ||
      tokens > kMaximumPositions - first_position) {
    return Status::InvalidArgument(
        "Qwen BF16 execution positions exceed the model context bound");
  }
  if (request_.generation == std::numeric_limits<std::uint64_t>::max()) {
    state_ = QwenBf16ExecutionState::kPoisoned;
    return Status::ResourceExhausted(
        "Qwen BF16 execution generation counter is exhausted");
  }

  request_ = {tokens, first_position, 1, request_.generation + 1};
  next_step_ = 0;
  state_ = QwenBf16ExecutionState::kRunning;
  return Status::Ok();
}

Status QwenBf16ExecutionCursor::RunNext(QwenBf16ExecutionDriver& driver) {
  if (state_ != QwenBf16ExecutionState::kRunning) {
    return Status::FailedPrecondition(
        "Qwen BF16 execution cursor has no runnable step");
  }
  const auto status = driver.Execute(schedule_[next_step_], request_);
  if (!status.ok()) {
    state_ = QwenBf16ExecutionState::kPoisoned;
    return status;
  }
  ++next_step_;
  if (next_step_ == schedule_.size()) {
    state_ = QwenBf16ExecutionState::kCompleted;
  }
  return Status::Ok();
}

}  // namespace pih
