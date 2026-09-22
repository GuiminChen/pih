#pragma once

#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/model/deepseek_recent_state_writer.h"

namespace pih {

class NvidiaDeepSeekRecentStateOperations final
    : public DeepSeekRecentStateOperations {
 public:
  static Result<NvidiaDeepSeekRecentStateOperations> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api);

  Status copy_d2d_async(std::uintptr_t destination,
                        std::uintptr_t source, std::size_t bytes,
                        std::uintptr_t stream) override;

 private:
  explicit NvidiaDeepSeekRecentStateOperations(
      std::uint64_t context_identity,
      const pih_nvidia_cuda_async_api_v1* async_api)
      : context_identity_(context_identity), async_api_(async_api) {}
  std::uint64_t context_identity_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
};

}  // namespace pih
