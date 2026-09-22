#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_tap_fixture_sequence.h"
#include "pih/model/qwen3_bf16_tap_suite_run.h"
#include "pih/model/qwen3_kv_sequence_lease.h"
#include "pih/model/qwen3_kv_semantic_observation_plan.h"
#include "pih/model/qwen3_semantic_token_ledger.h"

namespace pih {

struct QwenBf16TapFixtureExecutionReceipt final {
  QwenNumericalTapRunReceipt tap_run;
  QwenKvCompletionEvent completion;
  std::int64_t sampled_token;
};

class QwenBf16TapFixtureBackend {
 public:
  virtual ~QwenBf16TapFixtureBackend() = default;
  virtual Result<QwenBf16TapFixtureExecutionReceipt> execute_and_capture(
      const QwenBf16TapFixtureInput& input,
      const QwenNumericalTapPlan& taps,
      const QwenKvBlockTable& block_table,
      const QwenKvAppendPlan& append_plan,
      std::span<const QwenKvSlotState> projected_slot_states,
      std::uint64_t fixture_generation) = 0;
};

enum class QwenBf16TapSuiteExecutorState : std::uint8_t {
  kReady = 0,
  kRunning,
  kSealed,
  kPoisoned,
  kReleased,
};

class QwenBf16TapSuiteExecutor final {
 public:
  static Result<QwenBf16TapSuiteExecutor> Admit(
      QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
      std::uint32_t sequence_generation,
      const QwenBf16TapSuitePlan& suite,
      const QwenBf16TapFixtureSequence& sequence,
      std::uint64_t suite_generation,
      std::uint64_t first_fixture_generation,
      QwenBf16TapFixtureBackend& backend);

  QwenBf16TapSuiteExecutor(const QwenBf16TapSuiteExecutor&) = delete;
  QwenBf16TapSuiteExecutor& operator=(const QwenBf16TapSuiteExecutor&) = delete;
  QwenBf16TapSuiteExecutor(QwenBf16TapSuiteExecutor&&) noexcept = default;
  QwenBf16TapSuiteExecutor& operator=(QwenBf16TapSuiteExecutor&&) = delete;

  Result<QwenBf16TapSuiteRunReceipt> run();
  Status publish_token_semantics(QwenSemanticOutcomeRecorder& recorder);
  Result<QwenKvSemanticObservationPlan> make_kv_observation_plan(
      std::uint64_t backing_bytes) const;
  Status release();

  [[nodiscard]] QwenBf16TapSuiteExecutorState state() const noexcept {
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
  QwenBf16TapSuiteExecutor(QwenKvSequenceLease lease,
                          QwenKvSlotPool& pool,
                          QwenBf16TapSuitePlan suite,
                          QwenBf16TapFixtureSequence sequence,
                          QwenBf16TapSuiteRun run,
                          std::uint64_t first_fixture_generation,
                          QwenBf16TapFixtureBackend& backend)
      : lease_(std::move(lease)), pool_(&pool), suite_(std::move(suite)),
        sequence_(std::move(sequence)), run_(std::move(run)),
        first_fixture_generation_(first_fixture_generation),
        backend_(&backend) {}

  QwenKvSequenceLease lease_;
  QwenKvSlotPool* pool_;
  QwenBf16TapSuitePlan suite_;
  QwenBf16TapFixtureSequence sequence_;
  QwenBf16TapSuiteRun run_;
  std::uint64_t first_fixture_generation_;
  QwenBf16TapFixtureBackend* backend_;
  QwenSemanticTokenLedger token_ledger_;
  QwenKvCompletionEvent last_completion_{};
  QwenBf16TapSuiteExecutorState state_ =
      QwenBf16TapSuiteExecutorState::kReady;
};

}  // namespace pih
