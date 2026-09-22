#pragma once

#include "pih/model/deepseek_compressor_state_writer.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih {

class NvidiaDeepSeekCompressorStateOperations final
    : public DeepSeekCompressorStateOperations {
 public:
  static Result<NvidiaDeepSeekCompressorStateOperations> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api,
      const pih_deepseek_kernels_api_v1& kernels);

  Status validate_host_error(std::uint32_t* host_error) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status pooling(DeepSeekCompressorPoolingLaunch launch) override;
  Status projection(
      DeepSeekCompressorBf16ProjectionLaunch launch) override;
  Status copy_error_d2h_async(std::uint32_t* host,
                              std::uintptr_t device,
                              std::uintptr_t stream) override;

  [[nodiscard]] std::uint64_t context_identity() const noexcept {
    return context_identity_;
  }

 private:
  explicit NvidiaDeepSeekCompressorStateOperations(
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
