#include "pih/model/qwen3_semantic_paired_executor.h"

#include <algorithm>

namespace pih {

Result<QwenSemanticControlReceipt> QwenSemanticPairedExecutor::run(
    const QwenNumericalRunIdentity& identity,
    std::span<const std::int64_t> tokens,
    QwenSemanticPairedExecutionDriver& driver) {
  constexpr std::size_t kFixtureTokens = 4098;
  constexpr std::int64_t kVocabularySize = 151936;
  if (state_ != QwenSemanticPairedExecutorState::kReady) {
    state_ = QwenSemanticPairedExecutorState::kPoisoned;
    return Status::FailedPrecondition(
        "Qwen semantic paired executor is not reusable");
  }
  if (tokens.size() != kFixtureTokens ||
      std::any_of(tokens.begin(), tokens.end(), [](std::int64_t token) {
        return token < 0 || token >= kVocabularySize;
      })) {
    return Status::InvalidArgument(
        "Qwen semantic paired fixture requires 4098 valid token ids");
  }
  state_ = QwenSemanticPairedExecutorState::kRunning;

  auto instrumented = driver.run_instrumented(tokens);
  if (!instrumented.ok()) {
    state_ = QwenSemanticPairedExecutorState::kPoisoned;
    return instrumented.status();
  }

  auto control = driver.run_control(tokens);
  if (!control.ok()) {
    state_ = QwenSemanticPairedExecutorState::kPoisoned;
    return control.status();
  }

  auto receipt = QwenSemanticControlReceipt::Create(
      identity, instrumented->tap_suite, instrumented->semantic, *control);
  if (!receipt.ok()) {
    state_ = QwenSemanticPairedExecutorState::kPoisoned;
    return receipt.status();
  }
  state_ = QwenSemanticPairedExecutorState::kCompleted;
  return receipt;
}

}  // namespace pih
