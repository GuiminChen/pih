#pragma once

#include "pih/model/deepseek_sparse_attention_driver.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih {

class NvidiaDeepSeekSparseAttentionOperations final
    : public DeepSeekSparseAttentionOperations {
 public:
  static Result<NvidiaDeepSeekSparseAttentionOperations> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api,
      const pih_deepseek_kernels_api_v1& kernels);

  Status validate_host_staging(
      const DeepSeekSparseAttentionHostStaging& staging) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status copy_h2d_async(std::uintptr_t device, const void* host,
                        std::size_t bytes, std::uintptr_t stream) override;
  Status attention(DeepSeekSparseAttentionLaunch launch) override;
  Status copy_d2h_async(void* host, std::uintptr_t device,
                        std::size_t bytes, std::uintptr_t stream) override;

  [[nodiscard]] std::uint64_t context_identity() const noexcept {
    return context_identity_;
  }

 private:
  explicit NvidiaDeepSeekSparseAttentionOperations(
      std::uint64_t context_identity,
      const pih_nvidia_cuda_async_api_v1* async_api,
      const pih_deepseek_kernels_api_v1* kernels)
      : context_identity_(context_identity), async_api_(async_api),
        kernels_(kernels) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
  const pih_deepseek_kernels_api_v1* kernels_ = nullptr;
};

}  // namespace pih
