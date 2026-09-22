#pragma once

#include "pih/backend/cuda/deepseek_dspark_attention.h"
#include "pih/backend/cuda/deepseek_dspark_prefill.h"
#include "pih/model/deepseek_attention_projection_submission_assembler.h"
#include "pih/model/deepseek_bound_dspark_mtp_operator_backend.h"
#include "pih/model/deepseek_mhc_sequence_executor.h"

namespace pih {

class DeepSeekDsparkDecodeStageOperations {
 public:
  virtual ~DeepSeekDsparkDecodeStageOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status positions(DeepSeekDsparkPositionLaunch launch) = 0;
  virtual Status mhc_pre(DeepSeekMhcPreLaunch launch) = 0;
  virtual Status quant(DeepSeekFp8ActivationQuantLaunch launch) = 0;
  virtual Status gemm(DeepSeekFp8GemmLaunch launch) = 0;
  virtual Status rms(DeepSeekRmsNormLaunch launch) = 0;
  virtual Status head_rms(DeepSeekHeadRmsLaunch launch) = 0;
  virtual Status rotary(DeepSeekRotaryLaunch launch) = 0;
  virtual Status kv_simulate(DeepSeekKvFp8SimulateLaunch launch) = 0;
  virtual Status recent_store(DeepSeekDsparkRecentStoreLaunch launch) = 0;
  virtual Status attention(DeepSeekDsparkAttentionLaunch launch) = 0;
  virtual Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch launch) = 0;
  virtual Status mhc_post(DeepSeekMhcPostLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
  virtual Status record_event(std::uintptr_t event,
                              std::uintptr_t stream) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) = 0;
  virtual Status synchronize_stream(std::uintptr_t stream) = 0;
};

// One fail-stop instance owns one strong MTP stage.  Submission completion is
// acknowledged by poll(); the surrounding attention transaction owns the
// stream event that ultimately authorizes the tentative recent-state bank.
class DeepSeekDsparkDecodeStageOperation final
    : public DeepSeekDsparkMtpStageOperation {
 public:
  static Result<DeepSeekDsparkDecodeStageOperation> Create(
      DeepSeekDsparkStageId stage,
      DeepSeekDsparkDecodeStageOperations& operations,
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
  DeepSeekDsparkStageId stage_ = DeepSeekDsparkStageId::kMtp0;
  DeepSeekDsparkDecodeStageOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  std::uintptr_t active_stream_ = 0;
  std::uintptr_t active_event_ = 0;
  bool active_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
