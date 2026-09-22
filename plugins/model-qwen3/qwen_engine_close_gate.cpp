#include "pih/model/qwen_engine_close_gate.h"

namespace pih {

QwenEngineOperationLease::QwenEngineOperationLease(
    QwenEngineOperationLease&& other) noexcept
    : gate_(other.gate_) {
  other.gate_ = nullptr;
}

QwenEngineOperationLease::~QwenEngineOperationLease() {
  if (gate_ != nullptr) gate_->release();
}

Status QwenEngineCloseGate::require_open() const {
  if (state() != QwenEngineCloseState::kOpen)
    return Status::FailedPrecondition("Qwen engine epoch is closing or closed");
  return Status::Ok();
}

Result<QwenEngineOperationLease> QwenEngineCloseGate::enter() {
  std::lock_guard lock(wait_mutex_);
  if (state() != QwenEngineCloseState::kOpen)
    return Status::FailedPrecondition("Qwen engine epoch is closing or closed");
  ++active_operations_;
  return QwenEngineOperationLease(*this);
}

bool QwenEngineCloseGate::request_close() {
  std::lock_guard lock(wait_mutex_);
  auto expected = QwenEngineCloseState::kOpen;
  return state_.compare_exchange_strong(
      expected, QwenEngineCloseState::kClosing, std::memory_order_acq_rel,
      std::memory_order_acquire);
}

void QwenEngineCloseGate::wait_drained() {
  std::unique_lock lock(wait_mutex_);
  wait_condition_.wait(lock, [this] { return active_operations_ == 0; });
}

void QwenEngineCloseGate::finish_close(Status outcome) noexcept {
  std::lock_guard lock(wait_mutex_);
  auto expected = QwenEngineCloseState::kClosing;
  if (state_.compare_exchange_strong(expected, QwenEngineCloseState::kClosed,
                                     std::memory_order_release,
                                     std::memory_order_acquire)) {
    close_outcome_.emplace(std::move(outcome));
    wait_condition_.notify_all();
  }
}

void QwenEngineCloseGate::release() noexcept {
  std::lock_guard lock(wait_mutex_);
  if (active_operations_ == 0) return;
  --active_operations_;
  if (active_operations_ == 0) wait_condition_.notify_all();
}

Status QwenEngineCloseGate::wait_closed() {
  std::unique_lock lock(wait_mutex_);
  wait_condition_.wait(lock, [this] {
    return state() == QwenEngineCloseState::kClosed;
  });
  if (!close_outcome_.has_value())
    return Status::Internal("Qwen close outcome is unavailable");
  return *close_outcome_;
}

}  // namespace pih
