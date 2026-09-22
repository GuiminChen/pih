#include "pih/backend/cuda/nvidia_deepseek_expert_compute_lane.h"

namespace pih {

Result<std::unique_ptr<NvidiaDeepSeekExpertComputeLane>>
NvidiaDeepSeekExpertComputeLane::Create(
    RegisteredPinnedAllocator& pinned_allocator,
    DeepSeekExpertSlotTable slots, DeepSeekExpertComputeArena arena,
    std::uint32_t packed_token_count, std::uintptr_t source_hidden_bf16,
    std::uintptr_t accumulator_f32,
    const CudaRuntimeResourceIdentity& runtime
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
    , std::uintptr_t dspark_source_hidden_bf16
#endif
    , const pih_deepseek_kernels_api_v1& kernels,
    const pih_nvidia_cuda_async_api_v1& async_api) {
  if (runtime.context == 0 || runtime.stream == 0 ||
      runtime.deepseek_expert_event == 0 || runtime.device_ordinal < 0 ||
      runtime.rank == UINT32_MAX || runtime.worker_generation == 0) {
    return Status::InvalidArgument(
        "NVIDIA DeepSeek compute runtime identity is invalid");
  }
  auto operations_value = NvidiaDeepSeekExpertCudaOperations::Create(
      runtime.context, async_api, kernels);
  if (!operations_value.ok()) return operations_value.status();
  if (operations_value->context_identity() != runtime.context) {
    return Status::FailedPrecondition(
        "NVIDIA DeepSeek compute context differs from runtime owner");
  }
  auto operations = std::make_unique<NvidiaDeepSeekExpertCudaOperations>(
      std::move(*operations_value));
  auto lane_value = DeepSeekExpertComputeLane::Create(
      *operations, pinned_allocator, std::move(slots), arena,
      packed_token_count, source_hidden_bf16, accumulator_f32,
      runtime.stream, runtime.deepseek_expert_event, runtime.context
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      , dspark_source_hidden_bf16
#endif
      );
  if (!lane_value.ok()) return lane_value.status();
  auto lane = std::make_unique<DeepSeekExpertComputeLane>(
      std::move(*lane_value));
  return std::unique_ptr<NvidiaDeepSeekExpertComputeLane>(
      new NvidiaDeepSeekExpertComputeLane(
          std::move(operations), std::move(lane)));
}

}  // namespace pih
