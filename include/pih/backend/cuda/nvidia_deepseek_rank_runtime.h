#pragma once

#include <memory>

#include "pih/backend/cuda/nvidia_deepseek_attention_operation_set.h"
#include "pih/backend/cuda/nvidia_deepseek_plan_operation_set.h"
#include "pih/backend/cuda/nvidia_deepseek_h2d_runtime.h"
#include "pih/backend/cuda/nvidia_deepseek_request_input_copy_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_rope_table_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_rank_runtime_view.h"
#include "pih/backend/cuda/owned_cuda_runtime_resources.h"
#include "pih/contracts/nvidia_cuda_memory_v1.h"
#include "pih/contracts/nvidia_cuda_resources_v1.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/contracts/memory_host_spill_v1.h"

namespace pih {

class NvidiaDeepSeekRankRuntime final : public NvidiaDeepSeekRankRuntimeView {
 public:
  static Result<std::unique_ptr<NvidiaDeepSeekRankRuntime>> Create(
      std::int32_t device_ordinal, std::uint32_t rank,
      std::uint64_t worker_generation,
      const DeepSeekOptimizationPolicy& optimization_policy,
      const pih_nvidia_cuda_memory_api_v1& memory_api,
      const pih_nvidia_cuda_resources_api_v1& resources_api,
      const pih_nvidia_cuda_async_api_v1& async_api,
      const pih_deepseek_kernels_api_v1& kernels_api,
      const pih_memory_host_spill_api_v1* host_spill_api);

  NvidiaDeepSeekRankRuntime(const NvidiaDeepSeekRankRuntime&) = delete;
  NvidiaDeepSeekRankRuntime& operator=(const NvidiaDeepSeekRankRuntime&) =
      delete;

  Result<std::unique_ptr<RegisteredPinnedAllocator>>
  CreateSharedPinnedHostAllocator() override;
  RegisteredPinnedAllocator* host_spill_pinned_allocator()
      noexcept override {
    return host_spill_pinned_allocator_.get();
  }
  Result<std::unique_ptr<Allocator>> CreateDeviceAllocator() override;
  Allocator* host_spill_device_allocator() noexcept override {
    return host_spill_device_allocator_.get();
  }
  Result<std::unique_ptr<MemoryCopier>> CreateDeviceMemoryCopier() override;
  [[nodiscard]] const CudaRuntimeResourceIdentity& identity()
      const noexcept override {
    return runtime_->identity();
  }
  [[nodiscard]] CudaRuntimeResourceDriver& resource_driver()
      noexcept override {
    return runtime_->driver();
  }
  [[nodiscard]] DeepSeekH2dRuntime& h2d() noexcept override { return *h2d_; }
  Status activate() override;
  [[nodiscard]] DeepSeekAttentionRuntimeOperations attention_operations()
      noexcept override { return attention_operations_->borrow(); }
  [[nodiscard]] DeepSeekFixedStateBankOperations& fixed_state_operations()
      noexcept override { return attention_operations_->fixed_state(); }
  [[nodiscard]] DeepSeekEndpointSequenceOperations& endpoint_operations()
      noexcept override { return plan_operations_->endpoint(); }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] DeepSeekDsparkEmbedOperations* dspark_embed_operations()
      noexcept override { return plan_operations_->dspark_embed(); }
  [[nodiscard]] DeepSeekDsparkHeadOperations* dspark_head_operations()
      noexcept override { return plan_operations_->dspark_head(); }
  [[nodiscard]] DeepSeekDsparkPrefillStageOperations*
  dspark_prefill_operations() noexcept override {
    return plan_operations_->dspark_prefill();
  }
  [[nodiscard]] DeepSeekDsparkDecodeStageOperations*
  dspark_decode_operations() noexcept override {
    return plan_operations_->dspark_decode();
  }
  [[nodiscard]] DeepSeekDsparkMoeStageOperations* dspark_moe_operations()
      noexcept override { return plan_operations_->dspark_moe(); }
#endif
  [[nodiscard]] DeepSeekAttentionProjectionOperations&
  attention_projection_operations() noexcept override {
    return plan_operations_->attention_projection();
  }
  [[nodiscard]] DeepSeekAttentionOutputProjectionOperations&
  attention_output_projection_operations() noexcept override {
    return plan_operations_->attention_output_projection();
  }
  [[nodiscard]] DeepSeekMhcSequenceOperations& mhc_operations()
      noexcept override { return plan_operations_->mhc(); }
  [[nodiscard]] DeepSeekRopeTableOperations& rope_table_operations()
      noexcept override { return *rope_table_operations_; }
  Result<std::unique_ptr<DeepSeekExpertKernelLaneOwner>>
  CreateExpertKernelLane(
      RegisteredPinnedAllocator& pinned_allocator,
      DeepSeekExpertSlotTable slots, DeepSeekExpertComputeArena arena,
      std::uint32_t packed_token_count, std::uintptr_t source_hidden_bf16,
      std::uintptr_t accumulator_f32
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      , std::uintptr_t dspark_source_hidden_bf16
#endif
      ) override;
  Result<std::unique_ptr<DeepSeekExpertTransferLaneOwner>>
  CreateExpertTransferLane(
      DeepSeekExpertHostSource& source,
      std::vector<std::uintptr_t> slot_bases,
      std::uint32_t transfer_reservation_window) override;
  Result<std::unique_ptr<DeepSeekSharedExpertOperations>> CreateSharedExpertOperations() override;
  DeepSeekLearnedRouterOperations* learned_router_operations()
      noexcept override {
    return &plan_operations_->learned_router();
  }
  std::uintptr_t plan_compute_stream() const noexcept override {
    return identity().stream;
  }
  std::uintptr_t plan_completion_event() const noexcept override {
    return identity().event;
  }
  DeepSeekRequestInputCopyOperations* request_input_copy_operations()
      noexcept override { return request_input_copy_operations_.get(); }

 private:
  static Result<std::unique_ptr<NvidiaDeepSeekRankRuntime>> CreateImpl(
      std::int32_t device_ordinal, std::uint32_t rank,
      std::uint64_t worker_generation,
      const DeepSeekOptimizationPolicy& optimization_policy,
      const pih_nvidia_cuda_memory_api_v1* memory_api,
      const pih_nvidia_cuda_resources_api_v1* resources_api,
      const pih_nvidia_cuda_async_api_v1* async_api,
      const pih_deepseek_kernels_api_v1* kernels_api,
      const pih_memory_host_spill_api_v1* host_spill_api);
  NvidiaDeepSeekRankRuntime(
      std::unique_ptr<OwnedCudaRuntimeResources> runtime,
      std::unique_ptr<NvidiaDeepSeekH2dRuntime> h2d,
      std::unique_ptr<NvidiaDeepSeekAttentionOperationSet>
          attention_operations,
      std::unique_ptr<NvidiaDeepSeekPlanOperationSet> plan_operations,
      std::unique_ptr<NvidiaDeepSeekRequestInputCopyOperations>
          request_input_copy_operations,
      std::unique_ptr<NvidiaDeepSeekRopeTableOperations>
          rope_table_operations,
      std::unique_ptr<RegisteredPinnedAllocator> host_spill_pinned_allocator,
      std::unique_ptr<Allocator> host_spill_device_allocator,
      const pih_nvidia_cuda_memory_api_v1* memory_api,
      const pih_nvidia_cuda_async_api_v1* async_api,
      const pih_deepseek_kernels_api_v1* kernels_api,
      const pih_memory_host_spill_api_v1* host_spill_api)
      : runtime_(std::move(runtime)), h2d_(std::move(h2d)),
        attention_operations_(std::move(attention_operations)),
        plan_operations_(std::move(plan_operations)),
        request_input_copy_operations_(std::move(request_input_copy_operations)),
        rope_table_operations_(std::move(rope_table_operations)),
        host_spill_pinned_allocator_(
            std::move(host_spill_pinned_allocator)),
        host_spill_device_allocator_(std::move(host_spill_device_allocator)),
        memory_api_(memory_api), async_api_(async_api),
        kernels_api_(kernels_api), host_spill_api_(host_spill_api) {}

  // H2D borrows the context owned by runtime_ and is destroyed first.
  std::unique_ptr<OwnedCudaRuntimeResources> runtime_;
  std::unique_ptr<NvidiaDeepSeekH2dRuntime> h2d_;
  std::unique_ptr<NvidiaDeepSeekAttentionOperationSet> attention_operations_;
  std::unique_ptr<NvidiaDeepSeekPlanOperationSet> plan_operations_;
  std::unique_ptr<NvidiaDeepSeekRequestInputCopyOperations>
      request_input_copy_operations_;
  std::unique_ptr<NvidiaDeepSeekRopeTableOperations> rope_table_operations_;
  std::unique_ptr<RegisteredPinnedAllocator> host_spill_pinned_allocator_;
  std::unique_ptr<Allocator> host_spill_device_allocator_;
  const pih_nvidia_cuda_memory_api_v1* memory_api_ = nullptr;
  const pih_nvidia_cuda_async_api_v1* async_api_ = nullptr;
  const pih_deepseek_kernels_api_v1* kernels_api_ = nullptr;
  const pih_memory_host_spill_api_v1* host_spill_api_ = nullptr;
};

}  // namespace pih
