#pragma once

#include "pih/model/deepseek_expert_transfer_driver.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/contracts/memory_host_spill_v1.h"

namespace pih {

class NvidiaDeepSeekH2dRuntime final : public DeepSeekH2dRuntime {
 public:
  static Result<NvidiaDeepSeekH2dRuntime> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api);
  static Result<NvidiaDeepSeekH2dRuntime> Create(
      std::uintptr_t retained_context,
      const pih_memory_host_spill_api_v1& host_spill_api);
  Status copy_async(std::uintptr_t destination, std::uintptr_t source,
                    std::uint64_t bytes, std::uintptr_t stream) override;
  Status record_event(std::uintptr_t event, std::uintptr_t stream) override;
  Result<DeepSeekTransferEventStatus> query_event(
      std::uintptr_t event) override;

 private:
  NvidiaDeepSeekH2dRuntime(std::uintptr_t context,
                           const pih_nvidia_cuda_async_api_v1* async_api,
                           const pih_memory_host_spill_api_v1* host_spill_api)
      : context_(context), async_api_(async_api),
        host_spill_api_(host_spill_api) {}
  std::uintptr_t context_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
  const pih_memory_host_spill_api_v1* host_spill_api_ = nullptr;
};

}  // namespace pih
