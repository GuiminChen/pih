#pragma once

#include "pih/model/deepseek_attention_output_projection_coordinator.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih {
class NvidiaDeepSeekAttentionOutputProjectionOperations final
    : public DeepSeekAttentionOutputProjectionOperations {
 public:
  static Result<NvidiaDeepSeekAttentionOutputProjectionOperations> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api,
      const pih_deepseek_kernels_api_v1& kernels);
  Status validate_host_error(std::uint32_t* host_error) override;
  Status zero_u32_async(std::uintptr_t device, std::uintptr_t stream) override;
  Status rotary(DeepSeekRotaryLaunch launch) override;
  Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch launch) override;
  Status quant(DeepSeekFp8ActivationQuantLaunch launch) override;
  Status gemm(DeepSeekFp8GemmLaunch launch) override;
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t device,
                              std::uintptr_t stream) override;
 private:
  explicit NvidiaDeepSeekAttentionOutputProjectionOperations(
      std::uint64_t identity,
      const pih_nvidia_cuda_async_api_v1* async_api,
      const pih_deepseek_kernels_api_v1* kernels) noexcept
      : context_identity_(identity), async_api_(async_api), kernels_(kernels) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
  const pih_deepseek_kernels_api_v1* kernels_ = nullptr;
};
}  // namespace pih
