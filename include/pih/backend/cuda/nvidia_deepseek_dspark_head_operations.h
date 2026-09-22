#pragma once

#include "pih/model/deepseek_dspark_head_executor.h"

namespace pih {
class NvidiaDeepSeekDsparkHeadOperations final
    : public DeepSeekDsparkHeadOperations {
 public:
  static Result<NvidiaDeepSeekDsparkHeadOperations> Create();
  Status validate_host_error(std::uint32_t* host_error) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status hc_head(DeepSeekHcHeadLaunch launch) override;
  Status rms_norm(DeepSeekRmsNormLaunch launch) override;
  Status lm_head(DeepSeekLmHeadLaunch launch) override;
  Status markov(DeepSeekDsparkMarkovLaunch launch) override;
  Status argmax(DeepSeekArgmaxLaunch launch) override;
  Status confidence(DeepSeekDsparkConfidenceLaunch launch) override;
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t device,
                              std::uintptr_t stream) override;
 private:
  explicit NvidiaDeepSeekDsparkHeadOperations(
      std::uint64_t context_identity) noexcept
      : context_identity_(context_identity) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
};
}  // namespace pih
