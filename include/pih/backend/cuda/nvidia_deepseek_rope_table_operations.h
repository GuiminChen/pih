#pragma once

#include "pih/backend/cuda/deepseek_rope_table.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih {

class NvidiaDeepSeekRopeTableOperations final
    : public DeepSeekRopeTableOperations {
 public:
  explicit NvidiaDeepSeekRopeTableOperations(
      const pih_deepseek_kernels_api_v1& kernels,
      std::uintptr_t context_identity,
      const pih_nvidia_cuda_async_api_v1& async_api) noexcept
      : kernels_(&kernels), context_identity_(context_identity),
        async_api_(&async_api) {}
  Status initialize(DeepSeekRopeTableLaunch launch) override;
  Status synchronize(std::uintptr_t stream) override;

 private:
  const pih_deepseek_kernels_api_v1* kernels_;
  std::uintptr_t context_identity_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
};

}  // namespace pih
