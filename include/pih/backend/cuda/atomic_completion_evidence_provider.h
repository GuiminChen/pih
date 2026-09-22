#pragma once

#include <atomic>
#include <cstdint>

#include "pih/backend/cuda/completion_event_slot.h"

namespace pih {

class CompletionLastErrorProbe {
 public:
  virtual ~CompletionLastErrorProbe() = default;
  virtual Status require_clean_last_error() = 0;
};

class AtomicCompletionEvidenceProvider final
    : public CompletionEvidenceProvider {
 public:
  static Result<AtomicCompletionEvidenceProvider> Create(
      CompletionLastErrorProbe& last_error_probe,
      const std::atomic<std::uint32_t>& device_error_code,
      const std::atomic<bool>& engine_poisoned);

  Result<CompletionPublicationEvidence> collect() override;

 private:
  AtomicCompletionEvidenceProvider(
      CompletionLastErrorProbe& last_error_probe,
      const std::atomic<std::uint32_t>& device_error_code,
      const std::atomic<bool>& engine_poisoned) noexcept
      : last_error_probe_(&last_error_probe),
        device_error_code_(&device_error_code),
        engine_poisoned_(&engine_poisoned) {}

  CompletionLastErrorProbe* last_error_probe_;
  const std::atomic<std::uint32_t>* device_error_code_;
  const std::atomic<bool>* engine_poisoned_;
};

}  // namespace pih
