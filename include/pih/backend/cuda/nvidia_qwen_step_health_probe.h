#pragma once

#include "pih/backend/cuda/atomic_completion_evidence_provider.h"
#include "pih/model/qwen3_bf16_step_health.h"

namespace pih {

class NvidiaQwenStepHealthProbe final
    : public QwenBf16SubmitThreadHealthProbe {
 public:
  explicit NvidiaQwenStepHealthProbe(
      CompletionLastErrorProbe& event_driver)
      : event_driver_(&event_driver) {}

  Status require_clean_last_error() override {
    return event_driver_->require_clean_last_error();
  }

 private:
  CompletionLastErrorProbe* event_driver_;
};

}  // namespace pih
