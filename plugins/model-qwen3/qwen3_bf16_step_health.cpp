#include "pih/model/qwen3_bf16_step_health.h"

namespace pih {

Result<QwenBf16AtomicStepHealthProvider>
QwenBf16AtomicStepHealthProvider::Create(
    QwenBf16SubmitThreadHealthProbe& probe,
    const std::atomic<bool>& engine_poisoned) {
  return QwenBf16AtomicStepHealthProvider(probe, engine_poisoned);
}

Result<QwenBf16StepHealth> QwenBf16AtomicStepHealthProvider::collect() {
  const Status clean = probe_->require_clean_last_error();
  if (!clean.ok()) return clean;
  return QwenBf16StepHealth{
      true, engine_poisoned_->load(std::memory_order_acquire)};
}

Result<QwenBf16TapSnapshotEvidenceProvider>
QwenBf16TapSnapshotEvidenceProvider::Create(
    QwenBf16StepHealthProvider& health) {
  return QwenBf16TapSnapshotEvidenceProvider(health);
}

Result<CompletionPublicationEvidence>
QwenBf16TapSnapshotEvidenceProvider::collect() {
  auto health = health_->collect();
  if (!health.ok()) return health.status();
  return CompletionPublicationEvidence{
      health->submit_thread_last_error_clean, 0,
      health->engine_poisoned};
}

}  // namespace pih
