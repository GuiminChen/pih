#pragma once

#include <optional>

#include "pih/backend/cuda/deepseek_expert_accumulate.h"
#include "pih/backend/cuda/deepseek_expert_swiglu.h"
#include "pih/backend/cuda/deepseek_fp4_gemm.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"
#include "pih/backend/cuda/deepseek_router_bf16_gemm.h"
#include "pih/model/deepseek_bound_dspark_mtp_operator_backend.h"
#include "pih/model/deepseek_dspark_resident_subwave_executor.h"
#include "pih/model/deepseek_learned_router.h"
#include "pih/model/deepseek_mhc_sequence_executor.h"

namespace pih {

class DeepSeekDsparkMoeStageOperations {
 public:
  virtual ~DeepSeekDsparkMoeStageOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status zero_bytes_async(std::uintptr_t device,
                                  std::uint64_t bytes,
                                  std::uintptr_t stream) = 0;
  virtual Status mhc_pre(DeepSeekMhcPreLaunch launch) = 0;
  virtual Status router_gemm(DeepSeekRouterBf16GemmLaunch launch) = 0;
  virtual Status copy_d2h_async(void* host, std::uintptr_t device,
                                std::uint64_t bytes,
                                std::uintptr_t stream) = 0;
  virtual Status record_event(std::uintptr_t event,
                              std::uintptr_t stream) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) = 0;
  virtual Status quant(DeepSeekFp8ActivationQuantLaunch launch) = 0;
  virtual Status fp4_gemm(DeepSeekFp4GemmLaunch launch) = 0;
  virtual Status shared_swiglu(
      DeepSeekSharedExpertSwiGluLaunch launch) = 0;
  virtual Status finalize(DeepSeekExpertFinalizeLaunch launch) = 0;
  virtual Status mhc_post(DeepSeekMhcPostLaunch launch) = 0;
  virtual Status synchronize_stream(std::uintptr_t stream) = 0;
};

class DeepSeekDsparkMoeStageOperation final
    : public DeepSeekDsparkMtpStageOperation {
 public:
  static constexpr std::uint32_t kTokenCount = 5;

  static Result<DeepSeekDsparkMoeStageOperation> Create(
      DeepSeekDsparkStageId stage,
      DeepSeekDsparkMoeStageOperations& operations,
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
  enum class State : std::uint8_t {
    kIdle,
    kRouter,
    kExperts,
    kFinalize,
    kPoisoned,
  };

  Status poison(Status status);
  Status launch_finalize();
  DeepSeekDsparkStageId stage_ = DeepSeekDsparkStageId::kMtp0;
  DeepSeekDsparkMoeStageOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  const DeepSeekDsparkMtpStageResources* active_ = nullptr;
  std::optional<DeepSeekExpertSubwavePlan> route_plan_;
  std::optional<DeepSeekDsparkResidentSubwaveExecutor> executor_;
  State state_ = State::kIdle;
};

}  // namespace pih
