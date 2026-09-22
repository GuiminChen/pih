#pragma once

#include "pih/backend/cuda/atomic_completion_evidence_provider.h"

namespace pih {

class NvidiaCompletionEventDriver final : public CompletionEventDriver,
                                          public CompletionLastErrorProbe {
 public:
  static Result<NvidiaCompletionEventDriver> Create();

  Status record(DriverEventHandle event,
                DriverStreamHandle stream) override;
  Result<CudaEventQueryResult> query(DriverEventHandle event) override;
  Status require_clean_last_error() override;

  [[nodiscard]] std::uintptr_t context_identity() const noexcept {
    return context_identity_;
  }

 private:
  explicit NvidiaCompletionEventDriver(std::uintptr_t context_identity)
      : context_identity_(context_identity) {}
  Status require_current_context() const;

  std::uintptr_t context_identity_;
};

}  // namespace pih
