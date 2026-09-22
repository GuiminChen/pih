#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_request_runner.h"
#include "pih/model/qwen3_bf16_tap_fixture_sequence.h"
#include "pih/model/qwen3_kv_semantic_observation_plan.h"
#include "pih/model/qwen3_semantic_token_ledger.h"

namespace pih {

enum class QwenBf16SemanticControlExecutorState : std::uint8_t {
  kReady,
  kRunning,
  kSealed,
  kPoisoned,
  kReleased,
};

class QwenBf16SemanticControlExecutor final {
 public:
  static Result<QwenBf16SemanticControlExecutor> Admit(
      QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
      std::uint32_t sequence_generation,
      const QwenBf16TapFixtureSequence& sequence,
      QwenBf16SequenceBackend& backend,
      QwenBf16CompletionIdentityProvider& completion);

  QwenBf16SemanticControlExecutor(
      const QwenBf16SemanticControlExecutor&) = delete;
  QwenBf16SemanticControlExecutor& operator=(
      const QwenBf16SemanticControlExecutor&) = delete;
  QwenBf16SemanticControlExecutor(
      QwenBf16SemanticControlExecutor&&) noexcept = default;
  QwenBf16SemanticControlExecutor& operator=(
      QwenBf16SemanticControlExecutor&&) = delete;

  Status run();
  Status publish_token_semantics(QwenSemanticOutcomeRecorder& recorder);
  Result<QwenKvSemanticObservationPlan> make_kv_observation_plan(
      std::uint64_t backing_bytes) const;
  Status release();

  [[nodiscard]] QwenBf16SemanticControlExecutorState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint32_t committed_tokens() const noexcept {
    return lease_.block_table().descriptor().committed_tokens;
  }
  [[nodiscard]] std::span<const QwenKvBlockHandle> reserved_handles()
      const noexcept {
    return lease_.block_table().reserved_handles();
  }
  [[nodiscard]] QwenKvCompletionEvent last_completion_event() const noexcept {
    return last_completion_;
  }

 private:
  QwenBf16SemanticControlExecutor(
      QwenKvSequenceLease lease, QwenKvSlotPool& pool,
      QwenBf16TapFixtureSequence sequence, QwenBf16SequenceBackend& backend,
      QwenBf16CompletionIdentityProvider& completion)
      : lease_(std::move(lease)), pool_(&pool), sequence_(std::move(sequence)),
        backend_(&backend), completion_(&completion) {}

  QwenKvSequenceLease lease_;
  QwenKvSlotPool* pool_;
  QwenBf16TapFixtureSequence sequence_;
  QwenBf16SequenceBackend* backend_;
  QwenBf16CompletionIdentityProvider* completion_;
  QwenSemanticTokenLedger token_ledger_;
  QwenKvCompletionEvent last_completion_{};
  QwenBf16SemanticControlExecutorState state_ =
      QwenBf16SemanticControlExecutorState::kReady;
};

}  // namespace pih
