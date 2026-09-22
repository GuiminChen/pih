#include "pih/backend/cuda/nvidia_deepseek_rank_runtime.h"


#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "pih/backend/cuda/nvidia_deepseek_expert_compute_lane.h"
#include "pih/backend/cuda/nvidia_deepseek_shared_expert_operations.h"
#include "pih/model/deepseek_expert_transfer_lane.h"

namespace pih {
namespace {

bool ExactCString(const char* value, std::string_view expected) noexcept {
  if (value == nullptr) return false;
  const auto* terminator = static_cast<const char*>(
      std::memchr(value, '\0', expected.size() + 1));
  return terminator == value + expected.size() &&
         std::memcmp(value, expected.data(), expected.size()) == 0;
}

bool IsExactDeepSeekKernelPack(
    const pih_deepseek_kernels_api_v1& kernels) noexcept {
  const auto* identity = kernels.identity;
  if (identity == nullptr || identity->struct_size != sizeof(*identity) ||
      identity->abi_version != PIH_KERNEL_PACK_ABI_VERSION_V1 ||
      !ExactCString(identity->pack_version, "1.0.0")) return false;
  return (ExactCString(identity->pack_id, "pih.kernels.deepseek-v4.sm89") &&
          ExactCString(identity->pack_abi, "pih.deepseek-sm89-kernel-pack.v1") &&
          ExactCString(identity->architecture, "sm89")) ||
         (ExactCString(identity->pack_id, "pih.kernels.deepseek-v4.sm90") &&
          ExactCString(identity->pack_abi, "pih.deepseek-sm90-kernel-pack.v1") &&
          ExactCString(identity->architecture, "sm90"));
}

Status FromAbiStatus(const pih_status_v1& status) {
  if (!pih_status_is_valid_v1(&status)) {
    return Status::Internal("CUDA provider returned an invalid status");
  }
  const std::string message(status.message,
      std::find(status.message, status.message + sizeof(status.message), '\0'));
  switch (status.code) {
    case PIH_STATUS_OK_V1: return Status::Ok();
    case PIH_STATUS_INVALID_ARGUMENT_V1:
      return Status::InvalidArgument(message);
    case PIH_STATUS_FAILED_PRECONDITION_V1:
      return Status::FailedPrecondition(message);
    case PIH_STATUS_INTERNAL_V1: return Status::Internal(message);
    case PIH_STATUS_UNAVAILABLE_V1: return Status::Unavailable(message);
    case PIH_STATUS_RESOURCE_EXHAUSTED_V1:
      return Status::ResourceExhausted(message);
    case PIH_STATUS_DEADLINE_EXCEEDED_V1:
      return Status::DeadlineExceeded(message);
    default: return Status::Internal("CUDA provider status is unknown");
  }
}

pih_cuda_allocation_v1 AbiAllocation(const Allocation& allocation,
                                     std::int32_t ordinal,
                                     std::uint32_t kind) {
  return {sizeof(pih_cuda_allocation_v1),
          PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1,
          reinterpret_cast<std::uintptr_t>(allocation.data),
          allocation.bytes, allocation.alignment, allocation.generation,
          ordinal, kind};
}

bool ValidRequestedAlignment(std::uint64_t alignment) noexcept {
  return alignment != 0 && (alignment & (alignment - 1)) == 0 &&
         alignment <= 256;
}

bool ValidCapabilityAllocation(const pih_cuda_allocation_v1& value,
                               std::uint64_t bytes,
                               std::uint64_t alignment) noexcept {
  return value.bytes == bytes && value.generation != 0 &&
         value.alignment >= alignment &&
         (value.alignment & (value.alignment - 1)) == 0 &&
         ((bytes == 0 && value.address == 0) ||
          (bytes != 0 && value.address != 0 &&
           value.address % value.alignment == 0));
}

class CapabilityDeviceAllocator final : public Allocator {
 public:
  CapabilityDeviceAllocator(const pih_nvidia_cuda_memory_api_v1& api,
                            std::int32_t ordinal) : api_(&api), ordinal_(ordinal) {}
  CapabilityDeviceAllocator(const pih_memory_host_spill_api_v1& api,
                            std::int32_t ordinal)
      : spill_api_(&api), ordinal_(ordinal) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    if (!ValidRequestedAlignment(alignment)) {
      return Status::InvalidArgument(
          "CUDA capability alignment must be a power of two at most 256");
    }
    pih_cuda_allocation_v1 value{};
    value.struct_size = sizeof(value);
    value.abi_version = PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1;
    auto status = FromAbiStatus(spill_api_ != nullptr
        ? spill_api_->allocate_device_slot(
              spill_api_->context, ordinal_, bytes, alignment, &value)
        : api_->allocate_device(
              api_->context, ordinal_, bytes, alignment, &value));
    if (!status.ok()) return status;
    auto device = Device::Create(DeviceType::kCuda, value.device_ordinal);
    if (value.struct_size != sizeof(value) ||
        value.abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
        !device.ok() || value.memory_kind != PIH_CUDA_MEMORY_DEVICE_V1 ||
        value.device_ordinal != ordinal_ ||
        !ValidCapabilityAllocation(value, bytes, alignment)) {
      if (spill_api_ != nullptr) {
        (void)spill_api_->deallocate_device_slot(spill_api_->context, &value);
      } else {
        (void)api_->deallocate_device(api_->context, &value);
      }
      return Status::FailedPrecondition(
          "CUDA memory provider returned a foreign device allocation");
    }
    return Allocation{reinterpret_cast<void*>(value.address), value.bytes,
                      value.alignment, value.generation, *device};
  }
  void deallocate(Allocation allocation) noexcept override {
    const auto value = AbiAllocation(allocation, ordinal_,
                                     PIH_CUDA_MEMORY_DEVICE_V1);
    if (spill_api_ != nullptr) {
      (void)spill_api_->deallocate_device_slot(spill_api_->context, &value);
    } else {
      (void)api_->deallocate_device(api_->context, &value);
    }
  }
 private:
  const pih_nvidia_cuda_memory_api_v1* api_ = nullptr;
  const pih_memory_host_spill_api_v1* spill_api_ = nullptr;
  std::int32_t ordinal_;
};

class CapabilityPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  explicit CapabilityPinnedAllocator(
      const pih_nvidia_cuda_memory_api_v1& api) : api_(&api) {}
  explicit CapabilityPinnedAllocator(
      const pih_memory_host_spill_api_v1& api) : spill_api_(&api) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    if (!ValidRequestedAlignment(alignment)) {
      return Status::InvalidArgument(
          "pinned capability alignment must be a power of two at most 256");
    }
    pih_cuda_allocation_v1 value{};
    value.struct_size = sizeof(value);
    value.abi_version = PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1;
    auto status = FromAbiStatus(spill_api_ != nullptr
        ? spill_api_->allocate_pinned(
              spill_api_->context, bytes, alignment, &value)
        : api_->allocate_pinned_host(
              api_->context, -1, bytes, alignment, &value));
    if (!status.ok()) return status;
    if (value.struct_size != sizeof(value) ||
        value.abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
        value.memory_kind != PIH_CUDA_MEMORY_PINNED_HOST_V1 ||
        value.device_ordinal != -1 ||
        !ValidCapabilityAllocation(value, bytes, alignment)) {
      if (spill_api_ != nullptr) {
        (void)spill_api_->deallocate_pinned(spill_api_->context, &value);
      } else {
        (void)api_->deallocate_pinned_host(api_->context, &value);
      }
      return Status::FailedPrecondition(
          "CUDA memory provider returned a foreign pinned allocation");
    }
    return Allocation{reinterpret_cast<void*>(value.address), value.bytes,
                      value.alignment, value.generation, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    const auto value = AbiAllocation(allocation, -1,
                                     PIH_CUDA_MEMORY_PINNED_HOST_V1);
    if (spill_api_ != nullptr) {
      (void)spill_api_->deallocate_pinned(spill_api_->context, &value);
    } else {
      (void)api_->deallocate_pinned_host(api_->context, &value);
    }
  }
 private:
  const pih_nvidia_cuda_memory_api_v1* api_ = nullptr;
  const pih_memory_host_spill_api_v1* spill_api_ = nullptr;
};

class CapabilityMemoryCopier final : public MemoryCopier {
 public:
  CapabilityMemoryCopier(const pih_nvidia_cuda_memory_api_v1& api,
                         std::int32_t ordinal) : api_(&api), ordinal_(ordinal) {}
  Status copy(void* destination, Device destination_device,
              const void* source, Device source_device,
              std::uint64_t bytes) override {
    if (source_device != Device::Cpu() ||
        destination_device.type() != DeviceType::kCuda ||
        destination_device.index() != ordinal_) {
      return Status::InvalidArgument(
          "CUDA memory capability requires CPU to exact-device copies");
    }
    return FromAbiStatus(api_->copy_h2d(
        api_->context, ordinal_, reinterpret_cast<std::uintptr_t>(destination),
        source, bytes));
  }
 private:
  const pih_nvidia_cuda_memory_api_v1* api_;
  std::int32_t ordinal_;
};

class CapabilityRuntimeResourceDriver final
    : public CudaRuntimeResourceDriver {
 public:
  explicit CapabilityRuntimeResourceDriver(
      const pih_nvidia_cuda_resources_api_v1& api) noexcept : api_(&api) {}

  Result<std::uintptr_t> retain_primary_context(
      std::int32_t device_ordinal, std::uint32_t context_flags) override {
    std::uintptr_t context = 0;
    auto status = FromAbiStatus(api_->retain_primary_context(
        api_->context, device_ordinal, context_flags, &context));
    if (!status.ok()) return status;
    if (context == 0) {
      return Status::FailedPrecondition(
          "CUDA provider returned an invalid primary context");
    }
    return context;
  }
  Status bind_runtime(std::int32_t device_ordinal,
                      std::uintptr_t context) override {
    return FromAbiStatus(
        api_->bind_runtime(api_->context, device_ordinal, context));
  }
  Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t context) override {
    DriverStreamHandle stream = 0;
    auto status = FromAbiStatus(api_->create_nonblocking_stream(
        api_->context, context, &stream));
    if (!status.ok()) return status;
    if (stream == 0) {
      return Status::FailedPrecondition(
          "CUDA provider returned an invalid stream");
    }
    return stream;
  }
  Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t context) override {
    DriverEventHandle event = 0;
    auto status = FromAbiStatus(api_->create_disable_timing_event(
        api_->context, context, &event));
    if (!status.ok()) return status;
    if (event == 0) {
      return Status::FailedPrecondition(
          "CUDA provider returned an invalid event");
    }
    return event;
  }
  void destroy_event(DriverEventHandle event) noexcept override {
    (void)api_->destroy_event(api_->context, event);
  }
  void destroy_stream(DriverStreamHandle stream) noexcept override {
    (void)api_->destroy_stream(api_->context, stream);
  }
  void release_primary_context(std::int32_t device_ordinal,
                               std::uintptr_t context) noexcept override {
    (void)api_->release_primary_context(api_->context, device_ordinal,
                                        context);
  }

 private:
  const pih_nvidia_cuda_resources_api_v1* api_;
};

}  // namespace

Result<std::unique_ptr<RegisteredPinnedAllocator>>
NvidiaDeepSeekRankRuntime::CreateSharedPinnedHostAllocator() {
  if (memory_api_ == nullptr) {
    return Status::FailedPrecondition("CUDA memory capability is unavailable");
  }
  return std::unique_ptr<RegisteredPinnedAllocator>(
      new CapabilityPinnedAllocator(*memory_api_));
}

Result<std::unique_ptr<Allocator>>
NvidiaDeepSeekRankRuntime::CreateDeviceAllocator() {
  if (memory_api_ == nullptr) {
    return Status::FailedPrecondition("CUDA memory capability is unavailable");
  }
  return std::unique_ptr<Allocator>(
      new CapabilityDeviceAllocator(*memory_api_, identity().device_ordinal));
}

Result<std::unique_ptr<MemoryCopier>>
NvidiaDeepSeekRankRuntime::CreateDeviceMemoryCopier() {
  if (memory_api_ == nullptr) {
    return Status::FailedPrecondition("CUDA memory capability is unavailable");
  }
  return std::unique_ptr<MemoryCopier>(
      new CapabilityMemoryCopier(*memory_api_, identity().device_ordinal));
}

Result<std::unique_ptr<DeepSeekExpertKernelLaneOwner>>
NvidiaDeepSeekRankRuntime::CreateExpertKernelLane(
    RegisteredPinnedAllocator& pinned_allocator,
    DeepSeekExpertSlotTable slots, DeepSeekExpertComputeArena arena,
    std::uint32_t packed_token_count, std::uintptr_t source_hidden_bf16,
    std::uintptr_t accumulator_f32
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
    , std::uintptr_t dspark_source_hidden_bf16
#endif
    ) {
  auto lane = NvidiaDeepSeekExpertComputeLane::Create(
      pinned_allocator, std::move(slots), arena, packed_token_count,
      source_hidden_bf16, accumulator_f32, identity(),
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      dspark_source_hidden_bf16,
#endif
      *kernels_api_, *async_api_);
  if (!lane.ok()) return lane.status();
  return std::unique_ptr<DeepSeekExpertKernelLaneOwner>(std::move(*lane));
}

Result<std::unique_ptr<DeepSeekSharedExpertOperations>>
NvidiaDeepSeekRankRuntime::CreateSharedExpertOperations() {
  if (async_api_ == nullptr || kernels_api_ == nullptr) {
    return Status::FailedPrecondition("Shared experts require Backend async and Kernel Pack capabilities");
  }
  auto operations = NvidiaDeepSeekSharedExpertOperations::Create(
      identity().context, *async_api_, *kernels_api_);
  if (!operations.ok()) return operations.status();
  return std::unique_ptr<DeepSeekSharedExpertOperations>(
      new NvidiaDeepSeekSharedExpertOperations(std::move(*operations)));
}

Result<std::unique_ptr<DeepSeekExpertTransferLaneOwner>>
NvidiaDeepSeekRankRuntime::CreateExpertTransferLane(
    DeepSeekExpertHostSource& source,
    std::vector<std::uintptr_t> slot_bases,
    std::uint32_t transfer_reservation_window) {
  auto lane = DeepSeekExpertTransferLane::Create(
      source, h2d(), std::move(slot_bases), identity(), resource_driver(),
      transfer_reservation_window, host_spill_api_);
  if (!lane.ok()) return lane.status();
  return std::unique_ptr<DeepSeekExpertTransferLaneOwner>(
      new DeepSeekExpertTransferLane(std::move(*lane)));
}

Status NvidiaDeepSeekRankRuntime::activate() {
  return resource_driver().bind_runtime(identity().device_ordinal,
                                        identity().context);
}

Result<std::unique_ptr<NvidiaDeepSeekRankRuntime>>
NvidiaDeepSeekRankRuntime::Create(
    std::int32_t device_ordinal, std::uint32_t rank,
    std::uint64_t worker_generation,
    const DeepSeekOptimizationPolicy& optimization_policy,
    const pih_nvidia_cuda_memory_api_v1& memory_api,
    const pih_nvidia_cuda_resources_api_v1& resources_api,
    const pih_nvidia_cuda_async_api_v1& async_api,
    const pih_deepseek_kernels_api_v1& kernels_api,
    const pih_memory_host_spill_api_v1* host_spill_api) {
  return CreateImpl(device_ordinal, rank, worker_generation,
                    optimization_policy, &memory_api, &resources_api, &async_api,
                    &kernels_api, host_spill_api);
}


Result<std::unique_ptr<NvidiaDeepSeekRankRuntime>>
NvidiaDeepSeekRankRuntime::CreateImpl(
    std::int32_t device_ordinal, std::uint32_t rank,
    std::uint64_t worker_generation,
    const DeepSeekOptimizationPolicy& optimization_policy,
    const pih_nvidia_cuda_memory_api_v1* memory_api,
    const pih_nvidia_cuda_resources_api_v1* resources_api,
    const pih_nvidia_cuda_async_api_v1* async_api,
    const pih_deepseek_kernels_api_v1* kernels_api,
    const pih_memory_host_spill_api_v1* host_spill_api) {
  if (device_ordinal < 0 || rank == UINT32_MAX || worker_generation == 0 ||
      memory_api == nullptr || resources_api == nullptr ||
      async_api == nullptr || kernels_api == nullptr ||
      (memory_api != nullptr &&
       (memory_api->struct_size != sizeof(*memory_api) ||
        memory_api->contract_version !=
            PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
        memory_api->context == nullptr ||
        memory_api->allocate_device == nullptr ||
        memory_api->deallocate_device == nullptr ||
        memory_api->allocate_pinned_host == nullptr ||
        memory_api->deallocate_pinned_host == nullptr ||
        memory_api->copy_h2d == nullptr)) ||
      (resources_api != nullptr &&
       (resources_api->struct_size != sizeof(*resources_api) ||
        resources_api->contract_version !=
            PIH_NVIDIA_CUDA_RESOURCES_ABI_VERSION_V1 ||
        resources_api->context == nullptr ||
        resources_api->retain_primary_context == nullptr ||
        resources_api->bind_runtime == nullptr ||
        resources_api->create_nonblocking_stream == nullptr ||
        resources_api->create_disable_timing_event == nullptr ||
        resources_api->destroy_event == nullptr ||
        resources_api->destroy_stream == nullptr ||
        resources_api->release_primary_context == nullptr)) ||
      (async_api != nullptr &&
       (async_api->struct_size != sizeof(*async_api) ||
        async_api->contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
        async_api->context == nullptr ||
        async_api->activate_context == nullptr ||
        async_api->validate_pinned_host == nullptr ||
        async_api->copy_async == nullptr ||
        async_api->memset_async == nullptr ||
        async_api->record_event == nullptr ||
        async_api->query_event == nullptr ||
        async_api->synchronize_stream == nullptr)) ||
      (kernels_api != nullptr &&
       (kernels_api->struct_size != sizeof(*kernels_api) ||
        kernels_api->contract_version !=
            PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1 ||
        !IsExactDeepSeekKernelPack(*kernels_api) ||
        kernels_api->launch_rope_table == nullptr ||
        kernels_api->launch_rms_norm == nullptr ||
        kernels_api->launch_fp8_activation_quant == nullptr ||
        kernels_api->launch_fp8_gemm == nullptr ||
        kernels_api->launch_head_rms == nullptr ||
        kernels_api->launch_rotary == nullptr ||
        kernels_api->launch_kv_fp8_simulate == nullptr ||
        kernels_api->launch_grouped_fp8_gemm == nullptr ||
        kernels_api->launch_route_gather == nullptr ||
        kernels_api->launch_fp4_gemm == nullptr ||
        kernels_api->launch_expert_swiglu == nullptr ||
        kernels_api->launch_expert_accumulate == nullptr ||
        kernels_api->launch_embedding == nullptr ||
        kernels_api->launch_hc_head == nullptr ||
        kernels_api->launch_lm_head == nullptr ||
        kernels_api->launch_argmax == nullptr ||
        kernels_api->launch_stochastic_sample == nullptr ||
        kernels_api->launch_compressor_pooling == nullptr ||
        kernels_api->launch_compressor_projection == nullptr ||
        kernels_api->launch_compressor_store == nullptr ||
        kernels_api->launch_indexer_projection == nullptr ||
        kernels_api->launch_index_score == nullptr ||
        kernels_api->launch_sparse_attention == nullptr ||
        kernels_api->launch_router_gemm == nullptr ||
        kernels_api->launch_mhc_pre == nullptr ||
        kernels_api->launch_mhc_post == nullptr ||
        kernels_api->launch_mhc_target_tap == nullptr ||
        kernels_api->launch_shared_swiglu == nullptr ||
        kernels_api->launch_expert_finalize == nullptr)) ||
      (host_spill_api != nullptr &&
       (host_spill_api->struct_size != sizeof(*host_spill_api) ||
        host_spill_api->contract_version !=
            PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
        host_spill_api->context == nullptr ||
        host_spill_api->allocate_pinned == nullptr ||
        host_spill_api->deallocate_pinned == nullptr ||
        host_spill_api->copy_h2d_async == nullptr ||
        host_spill_api->record_event == nullptr ||
        host_spill_api->query_event == nullptr ||
        host_spill_api->create_completion_event == nullptr ||
        host_spill_api->destroy_completion_event == nullptr ||
        host_spill_api->allocate_device_slot == nullptr ||
        host_spill_api->deallocate_device_slot == nullptr ||
        host_spill_api->inspect == nullptr))) {
    return Status::InvalidArgument(
        "NVIDIA DeepSeek rank runtime identity is invalid");
  }
  std::unique_ptr<CudaRuntimeResourceDriver> resource_driver;
  if (resources_api != nullptr) {
    resource_driver =
        std::make_unique<CapabilityRuntimeResourceDriver>(*resources_api);
  } else {
    return Status::FailedPrecondition(
        "CUDA resource capability is unavailable");
  }
  auto runtime = OwnedCudaRuntimeResources::Create(
      device_ordinal, rank, worker_generation, PIH_CUDA_CONTEXT_SCHED_YIELD_V1,
      std::move(resource_driver));
  if (!runtime.ok()) return runtime.status();
  auto h2d = host_spill_api != nullptr
      ? NvidiaDeepSeekH2dRuntime::Create(runtime->identity().context,
                                        *host_spill_api)
      : NvidiaDeepSeekH2dRuntime::Create(runtime->identity().context,
                                        *async_api);
  if (!h2d.ok()) return h2d.status();
  auto attention_operations =
      NvidiaDeepSeekAttentionOperationSet::Create(
          *kernels_api, runtime->identity().context, *async_api);
  if (!attention_operations.ok()) return attention_operations.status();
  auto plan_operations =
      NvidiaDeepSeekPlanOperationSet::Create(
          optimization_policy, *kernels_api, runtime->identity().context,
          *async_api);
  if (!plan_operations.ok()) return plan_operations.status();
  auto request_input_copy_operations =
      NvidiaDeepSeekRequestInputCopyOperations::Create(
          runtime->identity().context, *async_api);
  if (!request_input_copy_operations.ok()) {
    return request_input_copy_operations.status();
  }
  if (runtime->identity().context == 0 ||
      runtime->identity().deepseek_paging_stream == 0 ||
      runtime->identity().deepseek_expert_event == 0) {
    return Status::FailedPrecondition(
        "NVIDIA DeepSeek rank runtime is missing mandatory resources");
  }
  std::unique_ptr<Allocator> host_spill_device_allocator;
  std::unique_ptr<RegisteredPinnedAllocator> host_spill_pinned_allocator;
  if (host_spill_api != nullptr) {
    host_spill_pinned_allocator =
        std::make_unique<CapabilityPinnedAllocator>(*host_spill_api);
    host_spill_device_allocator = std::make_unique<CapabilityDeviceAllocator>(
        *host_spill_api, device_ordinal);
  }
  const auto context_identity = runtime->identity().context;
  return std::unique_ptr<NvidiaDeepSeekRankRuntime>(
      new NvidiaDeepSeekRankRuntime(
          std::make_unique<OwnedCudaRuntimeResources>(std::move(*runtime)),
          std::make_unique<NvidiaDeepSeekH2dRuntime>(std::move(*h2d)),
          std::make_unique<NvidiaDeepSeekAttentionOperationSet>(
              std::move(*attention_operations)),
          std::make_unique<NvidiaDeepSeekPlanOperationSet>(
              std::move(*plan_operations)),
          std::make_unique<NvidiaDeepSeekRequestInputCopyOperations>(
              std::move(*request_input_copy_operations)),
          std::make_unique<NvidiaDeepSeekRopeTableOperations>(
              *kernels_api, context_identity, *async_api),
          std::move(host_spill_pinned_allocator),
          std::move(host_spill_device_allocator), memory_api, async_api,
          kernels_api, host_spill_api));
}

}  // namespace pih
