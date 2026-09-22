#pragma once

#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/model/deepseek_learned_router_coordinator.h"

namespace pih {

class NvidiaDeepSeekLearnedRouterOperations final
    : public DeepSeekLearnedRouterOperations {
 public:
  static Result<NvidiaDeepSeekLearnedRouterOperations> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api,
      const pih_deepseek_kernels_api_v1& kernels);
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status gemm(DeepSeekRouterBf16GemmLaunch launch) override;
  Status copy_d2h_async(void* host, std::uintptr_t device,
                        std::uint64_t bytes,
                        std::uintptr_t stream) override;
  Status record_event(std::uintptr_t event,
                      std::uintptr_t stream) override;
  Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) override;

 private:
  NvidiaDeepSeekLearnedRouterOperations(
      std::uint64_t context,
      const pih_nvidia_cuda_async_api_v1* async_api,
      const pih_deepseek_kernels_api_v1* kernels)
      : context_identity_(context), async_api_(async_api), kernels_(kernels) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
  const pih_deepseek_kernels_api_v1* kernels_ = nullptr;
};

}  // namespace pih
