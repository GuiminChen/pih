#pragma once

#include "pih/model/deepseek_dspark_gpu_state_digest.h"

namespace pih {

class NvidiaDeepSeekDsparkStateDigestOperations final
    : public DeepSeekDsparkGpuStateDigestOperations {
 public:
  static Result<NvidiaDeepSeekDsparkStateDigestOperations> Create();
  Status launch_component_sha256(
      const DeepSeekDsparkGpuStateDigestSubmission& submission) override;
  Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t completion_event) override;

 private:
  explicit NvidiaDeepSeekDsparkStateDigestOperations(std::uint64_t context)
      : context_identity_(context) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
};

}  // namespace pih
