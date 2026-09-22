#include "pih/model/qwen3_bf16_request_gate.h"

namespace pih {

Status QwenBf16RequestGate::begin() {
  auto expected = QwenBf16RequestGateState::kReady;
  if (state_.compare_exchange_strong(
          expected, QwenBf16RequestGateState::kRunning,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return Status::Ok();
  }
  if (expected == QwenBf16RequestGateState::kRunning) {
    return Status::ResourceExhausted(
        "Qwen single-request engine is already running");
  }
  return Status::FailedPrecondition("Qwen request gate is not ready");
}

Status QwenBf16RequestGate::finish(bool succeeded) {
  auto expected = QwenBf16RequestGateState::kRunning;
  const auto next = succeeded ? QwenBf16RequestGateState::kReady
                              : QwenBf16RequestGateState::kPoisoned;
  if (!state_.compare_exchange_strong(
          expected, next, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "Qwen request gate can finish only a running request");
  }
  return Status::Ok();
}

Status QwenBf16RequestGate::close() {
  auto current = state_.load(std::memory_order_acquire);
  while (true) {
    if (current == QwenBf16RequestGateState::kClosed) return Status::Ok();
    if (current == QwenBf16RequestGateState::kRunning) {
      return Status::FailedPrecondition(
          "Qwen request gate cannot close while running");
    }
    if (state_.compare_exchange_weak(
            current, QwenBf16RequestGateState::kClosed,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return Status::Ok();
    }
  }
}

}  // namespace pih
