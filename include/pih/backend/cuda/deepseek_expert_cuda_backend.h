#pragma once

#include <cstddef>
#include <cstdint>

#include "pih/backend/cuda/deepseek_expert_accumulate.h"
#include "pih/backend/cuda/deepseek_expert_swiglu.h"
#include "pih/backend/cuda/deepseek_fp4_gemm.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"
#include "pih/backend/cuda/deepseek_route_gather.h"
#include "pih/model/deepseek_expert_compute_driver.h"

namespace pih {

struct DeepSeekExpertHostStaging final {
  float* route_weights = nullptr;
  std::uint32_t* token_indices = nullptr;
  std::uint32_t* error_flag = nullptr;
  std::uint32_t capacity = 0;
};

class DeepSeekExpertCudaOperations {
 public:
  virtual ~DeepSeekExpertCudaOperations() = default;
  virtual Status validate_host_staging(
      const DeepSeekExpertHostStaging& staging) = 0;
  virtual Status zero_u32_async(std::uintptr_t device, std::uintptr_t stream) = 0;
  virtual Status copy_h2d_async(std::uintptr_t device, const void* host,
                                std::size_t bytes, std::uintptr_t stream) = 0;
  virtual Status copy_d2h_async(void* host, std::uintptr_t device,
                                std::size_t bytes, std::uintptr_t stream) = 0;
  virtual Status gather(DeepSeekRouteGatherLaunch launch) = 0;
  virtual Status quantize(DeepSeekFp8ActivationQuantLaunch launch) = 0;
  virtual Status gemm(DeepSeekFp4GemmLaunch launch) = 0;
  virtual Status swiglu(DeepSeekExpertSwiGluLaunch launch) = 0;
  virtual Status accumulate(DeepSeekExpertAccumulateLaunch launch) = 0;
  virtual Status record_event(std::uintptr_t event, std::uintptr_t stream) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t event) = 0;
};

class DeepSeekExpertCudaBackend final : public DeepSeekExpertComputeBackend {
 public:
  static Result<DeepSeekExpertCudaBackend> Create(
      DeepSeekExpertCudaOperations& operations,
      DeepSeekExpertHostStaging staging, std::uintptr_t completion_event,
      std::uint64_t context_identity);

  Status submit(const DeepSeekExpertComputeSubmission& submission) override;
  Result<DeepSeekExpertAsyncStatus> poll() override;

 private:
  DeepSeekExpertCudaOperations* operations_ = nullptr;
  DeepSeekExpertHostStaging staging_;
  std::uintptr_t completion_event_ = 0;
  std::uint64_t context_identity_ = 0;
  bool inflight_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
