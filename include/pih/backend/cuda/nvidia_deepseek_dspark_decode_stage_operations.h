#pragma once

#include "pih/model/deepseek_dspark_decode_stage_operation.h"

namespace pih {

class NvidiaDeepSeekDsparkDecodeStageOperations final
    : public DeepSeekDsparkDecodeStageOperations {
 public:
  static Result<NvidiaDeepSeekDsparkDecodeStageOperations> Create();

  Status validate_host_error(std::uint32_t* host_error) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status positions(DeepSeekDsparkPositionLaunch launch) override;
  Status mhc_pre(DeepSeekMhcPreLaunch launch) override;
  Status quant(DeepSeekFp8ActivationQuantLaunch launch) override;
  Status gemm(DeepSeekFp8GemmLaunch launch) override;
  Status rms(DeepSeekRmsNormLaunch launch) override;
  Status head_rms(DeepSeekHeadRmsLaunch launch) override;
  Status rotary(DeepSeekRotaryLaunch launch) override;
  Status kv_simulate(DeepSeekKvFp8SimulateLaunch launch) override;
  Status recent_store(DeepSeekDsparkRecentStoreLaunch launch) override;
  Status attention(DeepSeekDsparkAttentionLaunch launch) override;
  Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch launch) override;
  Status mhc_post(DeepSeekMhcPostLaunch launch) override;
  Status copy_error_d2h_async(std::uint32_t* host,
                              std::uintptr_t device,
                              std::uintptr_t stream) override;
  Status record_event(std::uintptr_t event,
                      std::uintptr_t stream) override;
  Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) override;
  Status synchronize_stream(std::uintptr_t stream) override;

 private:
  explicit NvidiaDeepSeekDsparkDecodeStageOperations(
      std::uint64_t context_identity) noexcept
      : context_identity_(context_identity) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
};

}  // namespace pih
