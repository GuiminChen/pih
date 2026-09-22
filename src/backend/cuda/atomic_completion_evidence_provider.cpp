#include "pih/backend/cuda/atomic_completion_evidence_provider.h"

namespace pih {

Result<AtomicCompletionEvidenceProvider>
AtomicCompletionEvidenceProvider::Create(
    CompletionLastErrorProbe& last_error_probe,
    const std::atomic<std::uint32_t>& device_error_code,
    const std::atomic<bool>& engine_poisoned) {
  return AtomicCompletionEvidenceProvider(last_error_probe, device_error_code,
                                          engine_poisoned);
}

Result<CompletionPublicationEvidence>
AtomicCompletionEvidenceProvider::collect() {
  const auto clean = last_error_probe_->require_clean_last_error();
  if (!clean.ok()) return clean;
  return CompletionPublicationEvidence{
      true, device_error_code_->load(std::memory_order_acquire),
      engine_poisoned_->load(std::memory_order_acquire)};
}

}  // namespace pih
