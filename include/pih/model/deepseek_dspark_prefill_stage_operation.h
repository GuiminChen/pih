#pragma once

#include "pih/backend/cuda/deepseek_attention_dense_primitives.h"
#include "pih/backend/cuda/deepseek_dspark_prefill.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"
#include "pih/backend/cuda/deepseek_fp8_gemm.h"
#include "pih/backend/cuda/deepseek_rms_norm.h"
#include "pih/model/deepseek_bound_dspark_mtp_operator_backend.h"

namespace pih {

class DeepSeekDsparkPrefillStageOperations {
 public:
  virtual ~DeepSeekDsparkPrefillStageOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status quant(DeepSeekFp8ActivationQuantLaunch launch) = 0;
  virtual Status gemm(DeepSeekFp8GemmLaunch launch) = 0;
  virtual Status rms(DeepSeekRmsNormLaunch launch) = 0;
  virtual Status rotary(DeepSeekRotaryLaunch launch) = 0;
  virtual Status kv_fp8_simulate(DeepSeekKvFp8SimulateLaunch launch) = 0;
  virtual Status recent_store(DeepSeekDsparkRecentStoreLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
  virtual Status record_event(std::uintptr_t event,
                              std::uintptr_t stream) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) = 0;
  virtual Status synchronize_stream(std::uintptr_t stream) = 0;
};

// Implements the attention-only official prefill path for one D-Spark stage.
// poll() acknowledges ordered stream submission; device completion and commit
// remain owned by the surrounding attention transaction's completion event.
class DeepSeekDsparkPrefillStageOperation final
    : public DeepSeekDsparkMtpStageOperation {
 public:
  static Result<DeepSeekDsparkPrefillStageOperation> Create(
      DeepSeekDsparkPrefillStageOperations& operations,
      std::uint32_t* host_error_flag);

  Status launch(
      const DeepSeekPipelinePlanDescriptor& plan,
      const DeepSeekDsparkMtpStageResources& resources) override;
  Status launch_prefill(
      const DeepSeekPipelinePlanDescriptor& plan,
      const DeepSeekDsparkMtpPrefillResources& resources) override;
  Result<DeepSeekStageComputeStatus> poll() override;
  Status cancel() override;

 private:
  DeepSeekDsparkPrefillStageOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  std::uintptr_t active_stream_ = 0;
  std::uintptr_t active_event_ = 0;
  bool active_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
