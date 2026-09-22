#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_kv_recycler.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih {

class NvidiaQwenBf16KvScrubDriver final : public QwenBf16KvScrubDriver {
 public:
  static Result<NvidiaQwenBf16KvScrubDriver> Create(
      const pih_nvidia_cuda_async_api_v1& async, std::uintptr_t context_identity,
      std::uint64_t timeout_ns);

  Status clear_and_wait(std::uintptr_t destination, std::uint64_t bytes,
                        DriverStreamHandle stream,
                        DriverEventHandle event) override;

 private:
  NvidiaQwenBf16KvScrubDriver(const pih_nvidia_cuda_async_api_v1& async,
                              std::uintptr_t context_identity,
                              std::uint64_t timeout_ns)
      : async_(&async), context_identity_(context_identity), timeout_ns_(timeout_ns) {}

  const pih_nvidia_cuda_async_api_v1* async_;
  std::uintptr_t context_identity_;
  std::uint64_t timeout_ns_;
};

}  // namespace pih
