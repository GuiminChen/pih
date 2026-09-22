#pragma once

#include <memory>

#include "pih/backend/cuda/cuda_runtime_resources.h"
#include "pih/backend/cuda/nvidia_deepseek_expert_cuda_operations.h"
#include "pih/model/deepseek_expert_lane_owner.h"
#include "pih/model/deepseek_expert_compute_lane.h"

namespace pih {

class NvidiaDeepSeekExpertComputeLane final
    : public DeepSeekExpertKernelLaneOwner {
 public:
  static Result<std::unique_ptr<NvidiaDeepSeekExpertComputeLane>> Create(
      RegisteredPinnedAllocator& pinned_allocator,
      DeepSeekExpertSlotTable slots, DeepSeekExpertComputeArena arena,
      std::uint32_t packed_token_count, std::uintptr_t source_hidden_bf16,
      std::uintptr_t accumulator_f32,
      const CudaRuntimeResourceIdentity& runtime
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      , std::uintptr_t dspark_source_hidden_bf16 = 0
#endif
      , const pih_deepseek_kernels_api_v1& kernels,
      const pih_nvidia_cuda_async_api_v1& async_api);

  NvidiaDeepSeekExpertComputeLane(const NvidiaDeepSeekExpertComputeLane&) = delete;
  NvidiaDeepSeekExpertComputeLane& operator=(const NvidiaDeepSeekExpertComputeLane&) = delete;

  [[nodiscard]] DeepSeekExpertKernelDriver& kernel() noexcept override {
    return lane_->kernel();
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] DeepSeekDsparkExpertKernelDriver* dspark_kernel()
      noexcept override {
    return lane_->dspark_kernel();
  }
#endif

 private:
  NvidiaDeepSeekExpertComputeLane(
      std::unique_ptr<NvidiaDeepSeekExpertCudaOperations> operations,
      std::unique_ptr<DeepSeekExpertComputeLane> lane)
      : operations_(std::move(operations)), lane_(std::move(lane)) {}

  std::unique_ptr<NvidiaDeepSeekExpertCudaOperations> operations_;
  std::unique_ptr<DeepSeekExpertComputeLane> lane_;
};

}  // namespace pih
