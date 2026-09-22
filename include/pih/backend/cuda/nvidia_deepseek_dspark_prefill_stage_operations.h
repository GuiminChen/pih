#pragma once

#include "pih/model/deepseek_dspark_prefill_stage_operation.h"

namespace pih {

class NvidiaDeepSeekDsparkPrefillStageOperations final
    : public DeepSeekDsparkPrefillStageOperations {
 public:
  static Result<NvidiaDeepSeekDsparkPrefillStageOperations> Create();

  Status validate_host_error(std::uint32_t* host_error) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status quant(DeepSeekFp8ActivationQuantLaunch launch) override;
  Status gemm(DeepSeekFp8GemmLaunch launch) override;
  Status rms(DeepSeekRmsNormLaunch launch) override;
  Status rotary(DeepSeekRotaryLaunch launch) override;
  Status kv_fp8_simulate(DeepSeekKvFp8SimulateLaunch launch) override;
  Status recent_store(DeepSeekDsparkRecentStoreLaunch launch) override;
  Status copy_error_d2h_async(std::uint32_t* host,
                              std::uintptr_t device,
                              std::uintptr_t stream) override;
  Status record_event(std::uintptr_t event,
                      std::uintptr_t stream) override;
  Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) override;
  Status synchronize_stream(std::uintptr_t stream) override;

 private:
  explicit NvidiaDeepSeekDsparkPrefillStageOperations(
      std::uint64_t context_identity) noexcept
      : context_identity_(context_identity) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
};

}  // namespace pih
