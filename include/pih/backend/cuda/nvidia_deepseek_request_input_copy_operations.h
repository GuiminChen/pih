#pragma once

#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/model/deepseek_request_input_staging_resources.h"

namespace pih {

class NvidiaDeepSeekRequestInputCopyOperations final
    : public DeepSeekRequestInputCopyOperations {
 public:
  static Result<NvidiaDeepSeekRequestInputCopyOperations> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api);

  Status copy_h2d_async(std::uintptr_t destination, const void* source,
                        std::uint64_t bytes,
                        std::uintptr_t stream) override;
  Status synchronize(std::uintptr_t stream) override;

 private:
  explicit NvidiaDeepSeekRequestInputCopyOperations(
      std::uint64_t context_identity,
      const pih_nvidia_cuda_async_api_v1* async_api) noexcept
      : context_identity_(context_identity), async_api_(async_api) {}

  std::uint64_t context_identity_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
};

}  // namespace pih
