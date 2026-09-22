#pragma once

#include "pih/model/deepseek_dspark_moe_stage_operation.h"

namespace pih {

class NvidiaDeepSeekDsparkMoeStageOperations final
    : public DeepSeekDsparkMoeStageOperations {
 public:
  static Result<NvidiaDeepSeekDsparkMoeStageOperations> Create();

  Status validate_host_error(std::uint32_t* host_error) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status zero_bytes_async(std::uintptr_t device, std::uint64_t bytes,
                          std::uintptr_t stream) override;
  Status mhc_pre(DeepSeekMhcPreLaunch launch) override;
  Status router_gemm(DeepSeekRouterBf16GemmLaunch launch) override;
  Status copy_d2h_async(void* host, std::uintptr_t device,
                        std::uint64_t bytes,
                        std::uintptr_t stream) override;
  Status record_event(std::uintptr_t event,
                      std::uintptr_t stream) override;
  Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) override;
  Status quant(DeepSeekFp8ActivationQuantLaunch launch) override;
  Status fp4_gemm(DeepSeekFp4GemmLaunch launch) override;
  Status shared_swiglu(
      DeepSeekSharedExpertSwiGluLaunch launch) override;
  Status finalize(DeepSeekExpertFinalizeLaunch launch) override;
  Status mhc_post(DeepSeekMhcPostLaunch launch) override;
  Status synchronize_stream(std::uintptr_t stream) override;

 private:
  explicit NvidiaDeepSeekDsparkMoeStageOperations(
      std::uint64_t context_identity) noexcept
      : context_identity_(context_identity) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
};

}  // namespace pih
