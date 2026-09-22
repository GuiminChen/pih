#pragma once

#include <atomic>

#include "pih/model/qwen3_bf16_step_transaction.h"

namespace pih {

class QwenBf16SubmitThreadHealthProbe {
 public:
  virtual ~QwenBf16SubmitThreadHealthProbe() = default;
  virtual Status require_clean_last_error() = 0;
};

class QwenBf16AtomicStepHealthProvider final
    : public QwenBf16StepHealthProvider {
 public:
  static Result<QwenBf16AtomicStepHealthProvider> Create(
      QwenBf16SubmitThreadHealthProbe& probe,
      const std::atomic<bool>& engine_poisoned);

  Result<QwenBf16StepHealth> collect() override;

 private:
  QwenBf16AtomicStepHealthProvider(
      QwenBf16SubmitThreadHealthProbe& probe,
      const std::atomic<bool>& engine_poisoned)
      : probe_(&probe), engine_poisoned_(&engine_poisoned) {}

  QwenBf16SubmitThreadHealthProbe* probe_;
  const std::atomic<bool>* engine_poisoned_;
};

class QwenBf16TapSnapshotEvidenceProvider final
    : public CompletionEvidenceProvider {
 public:
  static Result<QwenBf16TapSnapshotEvidenceProvider> Create(
      QwenBf16StepHealthProvider& health);
  Result<CompletionPublicationEvidence> collect() override;

 private:
  explicit QwenBf16TapSnapshotEvidenceProvider(
      QwenBf16StepHealthProvider& health)
      : health_(&health) {}
  QwenBf16StepHealthProvider* health_;
};

}  // namespace pih
