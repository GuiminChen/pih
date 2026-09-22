#pragma once

#include "pih/model/deepseek_mhc_sequence_executor.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"

namespace pih {

class NvidiaDeepSeekMhcBranchDriver {
 public:
  virtual ~NvidiaDeepSeekMhcBranchDriver() = default;
  virtual Status launch(DeepSeekMhcBranchLaunch launch) = 0;
};

class NvidiaDeepSeekMhcSequenceOperations final
    : public DeepSeekMhcSequenceOperations {
 public:
  static Result<NvidiaDeepSeekMhcSequenceOperations> Create(
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api,
      NvidiaDeepSeekMhcBranchDriver& branch,
      const pih_deepseek_kernels_api_v1& kernels);

  Status validate_host_error(std::uint32_t* host_error) override;
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override;
  Status pre(DeepSeekMhcPreLaunch launch) override;
  Status branch(DeepSeekMhcBranchLaunch launch) override;
  Status post(DeepSeekMhcPostLaunch launch) override;
  Status target_hidden_tap(
      DeepSeekMhcTargetHiddenTapLaunch launch) override;
  Status copy_error_d2h_async(std::uint32_t* host,
                              std::uintptr_t device,
                              std::uintptr_t stream) override;

 private:
  NvidiaDeepSeekMhcSequenceOperations(
      std::uint64_t context_identity,
      const pih_nvidia_cuda_async_api_v1* async_api,
      NvidiaDeepSeekMhcBranchDriver& branch,
      const pih_deepseek_kernels_api_v1* kernels) noexcept
      : context_identity_(context_identity), async_api_(async_api),
        branch_(&branch), kernels_(kernels) {}
  Status require_context() const;
  std::uint64_t context_identity_ = 0;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
  NvidiaDeepSeekMhcBranchDriver* branch_ = nullptr;
  const pih_deepseek_kernels_api_v1* kernels_ = nullptr;
};

}  // namespace pih
