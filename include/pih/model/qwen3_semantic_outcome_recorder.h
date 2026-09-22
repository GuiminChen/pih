#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_semantic_control.h"

namespace pih {

enum class QwenSemanticOutcomeRecorderState : std::uint8_t {
  kCollecting,
  kSealed,
  kPoisoned,
};

class QwenSemanticOutcomeRecorder final {
 public:
  static constexpr std::uint64_t kMaximumDispatchBytes = 1ULL << 20;
  static constexpr std::uint64_t kMaximumLogitsBytes = 1ULL << 20;
  static constexpr std::uint64_t kMaximumTrajectoryBytes = 1ULL << 20;
  static constexpr std::uint64_t kMaximumKvStateBytes = 1ULL << 30;
  static constexpr std::uint64_t kMaximumLedgerBytes = 1ULL << 20;

  Status record_dispatch(std::span<const std::byte> bytes);
  Status record_dispatch(const QwenBf16CommandBuffer& commands);
  Status record_final_logits(std::span<const std::byte> bytes);
  Status record_greedy_trajectory(std::span<const std::byte> bytes);
  Status record_kv_state(std::span<const std::byte> bytes);
  Status record_accepted_token_ledger(std::span<const std::byte> bytes);
  Status observe_capacity(std::uint64_t device_bytes,
                          std::uint64_t pinned_bytes);
  Result<QwenSemanticOutcome> seal();

  [[nodiscard]] QwenSemanticOutcomeRecorderState state() const noexcept {
    return state_;
  }

 private:
  Status record(std::span<const std::byte> bytes, std::uint64_t maximum,
                const char* domain, Sha256Digest& destination, bool& present);
  Status poison(const char* message);

  QwenSemanticOutcome outcome_{};
  bool dispatch_present_ = false;
  bool logits_present_ = false;
  bool trajectory_present_ = false;
  bool kv_present_ = false;
  bool ledger_present_ = false;
  QwenSemanticOutcomeRecorderState state_ =
      QwenSemanticOutcomeRecorderState::kCollecting;
};

}  // namespace pih
