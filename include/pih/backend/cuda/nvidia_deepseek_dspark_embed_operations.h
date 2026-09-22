#pragma once

#include "pih/model/deepseek_dspark_embed_coordinator.h"

namespace pih {
class NvidiaDeepSeekDsparkEmbedOperations final
    : public DeepSeekDsparkEmbedOperations {
 public:
  static Result<NvidiaDeepSeekDsparkEmbedOperations> Create();
  Status validate_host_error(std::uint32_t* host_error) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status quant(DeepSeekFp8ActivationQuantLaunch launch) override;
  Status gemm(DeepSeekFp8GemmLaunch launch) override;
  Status rms(DeepSeekRmsNormLaunch launch) override;
  Status draft_init(DeepSeekDsparkDraftInitLaunch launch) override;
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t device,
                              std::uintptr_t stream) override;
 private:
  explicit NvidiaDeepSeekDsparkEmbedOperations(
      std::uint64_t context_identity) noexcept
      : context_identity_(context_identity) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
};
}  // namespace pih
