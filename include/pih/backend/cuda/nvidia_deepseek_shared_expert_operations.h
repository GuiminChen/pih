#pragma once

#include "pih/model/deepseek_shared_expert_driver.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih {

class NvidiaDeepSeekSharedExpertOperations final
    : public DeepSeekSharedExpertOperations {
 public:
  // Both capability tables and their owning plugins must outlive this adapter.
  static Result<NvidiaDeepSeekSharedExpertOperations> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api,
      const pih_deepseek_kernels_api_v1& kernels);

  Status validate_host_error(std::uint32_t* host) override;
  Status prepare_moe(std::uintptr_t layer_input_bf16,
      std::uintptr_t routed_input_bf16, std::uintptr_t accumulator_f32,
      std::uint32_t token_count, std::uintptr_t stream) override;
  Status copy_error_d2h_async(std::uint32_t* host,
      std::uintptr_t device, std::uintptr_t stream) override;
  Status record_event(std::uintptr_t event, std::uintptr_t stream) override;
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t event) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status quantize(DeepSeekFp8ActivationQuantLaunch launch) override;
  Status gemm(DeepSeekFp8GemmLaunch launch) override;
  Status swiglu(DeepSeekSharedExpertSwiGluLaunch launch) override;
  Status finalize(DeepSeekExpertFinalizeLaunch launch) override;

  [[nodiscard]] std::uint64_t context_identity() const noexcept {
    return context_identity_;
  }

 private:
  NvidiaDeepSeekSharedExpertOperations(
      std::uint64_t context_identity,
      const pih_nvidia_cuda_async_api_v1* async_api,
      const pih_deepseek_kernels_api_v1* kernels)
      : context_identity_(context_identity), async_api_(async_api), kernels_(kernels) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
  const pih_deepseek_kernels_api_v1* kernels_ = nullptr;
};

}  // namespace pih
