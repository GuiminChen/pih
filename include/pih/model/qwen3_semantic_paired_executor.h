#pragma once

#include <cstdint>
#include <span>

#include "pih/model/qwen3_semantic_outcome_recorder.h"

namespace pih {

struct QwenSemanticInstrumentedRun final {
  QwenBf16TapSuiteRunReceipt tap_suite;
  QwenSemanticOutcome semantic;
};

class QwenSemanticPairedExecutionDriver {
 public:
  virtual ~QwenSemanticPairedExecutionDriver() = default;
  virtual Result<QwenSemanticInstrumentedRun> run_instrumented(
      std::span<const std::int64_t> tokens) = 0;
  virtual Result<QwenSemanticOutcome> run_control(
      std::span<const std::int64_t> tokens) = 0;
};

enum class QwenSemanticPairedExecutorState : std::uint8_t {
  kReady,
  kRunning,
  kCompleted,
  kPoisoned,
};

class QwenSemanticPairedExecutor final {
 public:
  Result<QwenSemanticControlReceipt> run(
      const QwenNumericalRunIdentity& identity,
      std::span<const std::int64_t> tokens,
      QwenSemanticPairedExecutionDriver& driver);

  [[nodiscard]] QwenSemanticPairedExecutorState state() const noexcept {
    return state_;
  }

 private:
  QwenSemanticPairedExecutorState state_ =
      QwenSemanticPairedExecutorState::kReady;
};

}  // namespace pih
