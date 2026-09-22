#include "pih/contracts/nvidia_cuda_v1.h"
#include "pih/contracts/nvidia_cuda_memory_v1.h"
#include "pih/contracts/nvidia_cuda_resources_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/contracts/platform_linux_v1.h"
#include "pih/plugin_sdk/abi.h"
#include "pih/backend/cuda/cuda_allocator.h"
#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_memory_copier.h"
#include "pih/backend/cuda/nvidia_pinned_host_allocator.h"
#include "pih/backend/cuda/nvidia_runtime_resource_driver.h"
#include "pih/backend/cuda/cuda_status.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

namespace {

static_assert(PIH_CUDA_CONTEXT_SCHED_YIELD_V1 == CU_CTX_SCHED_YIELD);

struct State final {
  struct AllocationRecord final {
    uint64_t bytes{};
    uint64_t alignment{};
    uint64_t generation{};
    int32_t device_ordinal{};
    uint32_t memory_kind{};
  };
  uint32_t phase{};
  const pih_host_api_v1* host{};
  const pih_platform_linux_api_v1* platform{};
  std::mutex memory_mutex;
  std::unordered_map<int32_t, std::unique_ptr<pih::CudaAllocator>>
      device_allocators;
  std::unordered_map<int32_t, std::unique_ptr<pih::CudaMemoryCopier>> copiers;
  std::unique_ptr<pih::NvidiaPinnedHostAllocator> pinned_allocator;
  std::unordered_map<uintptr_t, AllocationRecord> allocations;
  pih::NvidiaRuntimeResourceDriver runtime_driver;
  int32_t prepared_device_ordinal{-1};
  uint32_t prepared_sm_major{};
  uint32_t prepared_sm_minor{};
  std::unordered_map<uintptr_t, std::pair<int32_t, uint32_t>> contexts;
  std::unordered_map<uintptr_t, uintptr_t> streams;
  std::unordered_map<uintptr_t, uintptr_t> events;
  std::atomic_bool resource_ownership_corrupted{};
};

pih_status_v1 Status(uint32_t code, const char* message = "") {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  std::strncpy(value.message, message, sizeof(value.message) - 1);
  return value;
}

pih_status_v1 CheckedCapabilityStatus(pih_status_v1 status,
                                      const char* invalid_message) {
  return pih_status_is_valid_v1(&status)
      ? status
      : Status(PIH_STATUS_INTERNAL_V1, invalid_message);
}

template <typename Operation>
pih_status_v1 ContainAbi(Operation operation, const char* failure) noexcept {
  try {
    return operation();
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "cuda_provider_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, failure);
  }
}

pih_status_v1 Advance(void* context, uint32_t expected) {
  auto& state = *static_cast<State*>(context);
  if (state.phase != expected) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  ++state.phase;
  return Status(PIH_STATUS_OK_V1);
}

State state;

bool IsReady(const State& state) {
  return state.phase == 4 && !state.resource_ownership_corrupted;
}

bool IsCleanupAllowed(const State& state) {
  return state.phase == 4 || state.phase == 5;
}

bool RollbackPrimaryContext(int32_t device_ordinal,
                            uintptr_t retained_context) noexcept {
  CUdevice device{};
  if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS) return false;
  bool clean = true;
  CUcontext current = nullptr;
  const auto current_status = cuCtxGetCurrent(&current);
  if (current_status != CUDA_SUCCESS) {
    clean = false;
  } else if (reinterpret_cast<uintptr_t>(current) == retained_context &&
             cuCtxSetCurrent(nullptr) != CUDA_SUCCESS) {
    clean = false;
  }
  if (cuDevicePrimaryCtxRelease(device) != CUDA_SUCCESS) clean = false;
  return clean;
}

pih_status_v1 FromStatus(const pih::Status& status) {
  if (status.ok()) return Status(PIH_STATUS_OK_V1);
  uint32_t code = PIH_STATUS_INTERNAL_V1;
  switch (status.code()) {
    case pih::StatusCode::kOk: code = PIH_STATUS_OK_V1; break;
    case pih::StatusCode::kInvalidArgument:
      code = PIH_STATUS_INVALID_ARGUMENT_V1; break;
    case pih::StatusCode::kFailedPrecondition:
      code = PIH_STATUS_FAILED_PRECONDITION_V1; break;
    case pih::StatusCode::kInternal: code = PIH_STATUS_INTERNAL_V1; break;
    case pih::StatusCode::kUnavailable: code = PIH_STATUS_UNAVAILABLE_V1; break;
    case pih::StatusCode::kResourceExhausted:
      code = PIH_STATUS_RESOURCE_EXHAUSTED_V1; break;
    case pih::StatusCode::kDeadlineExceeded:
      code = PIH_STATUS_DEADLINE_EXCEEDED_V1; break;
  }
  const auto message = status.message();
  pih_status_v1 result = Status(code);
  const auto bytes = std::min(message.size(), sizeof(result.message) - 1);
  std::memcpy(result.message, message.data(), bytes);
  return result;
}

bool ValidOutput(pih_cuda_allocation_v1* allocation) {
  return allocation != nullptr &&
         allocation->struct_size == sizeof(*allocation) &&
         allocation->abi_version == PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 &&
         allocation->address == 0 && allocation->bytes == 0 &&
         allocation->generation == 0;
}

bool OwnsDeviceRange(const State& state, uintptr_t address, uint64_t bytes,
                     int32_t device_ordinal) {
  if (address == 0 || bytes == 0) return false;
  for (const auto& [base, record] : state.allocations) {
    if (record.memory_kind != PIH_CUDA_MEMORY_DEVICE_V1 ||
        record.device_ordinal != device_ordinal || address < base) {
      continue;
    }
    const auto offset = address - base;
    if (offset <= record.bytes && bytes <= record.bytes - offset) return true;
  }
  return false;
}

bool OwnsPinnedRange(const State& state, uintptr_t address, uint64_t bytes) {
  if (address == 0 || bytes == 0) return false;
  for (const auto& [base, record] : state.allocations) {
    if (record.memory_kind != PIH_CUDA_MEMORY_PINNED_HOST_V1 ||
        record.device_ordinal != -1 || address < base) {
      continue;
    }
    const auto offset = address - base;
    if (offset <= record.bytes && bytes <= record.bytes - offset) return true;
  }
  return false;
}

void ExportAllocation(const pih::Allocation& source, uint32_t kind,
                      int32_t device_ordinal,
                      pih_cuda_allocation_v1& destination) {
  destination.address = reinterpret_cast<uintptr_t>(source.data);
  destination.bytes = source.bytes;
  destination.alignment = source.alignment;
  destination.generation = source.generation;
  destination.device_ordinal = device_ordinal;
  destination.memory_kind = kind;
}

pih_status_v1 AllocateDevice(void* context, int32_t device_ordinal,
                             uint64_t bytes, uint64_t alignment,
                             pih_cuda_allocation_v1* allocation) {
  if (context == nullptr || device_ordinal < 0 || !ValidOutput(allocation)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_allocation_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (state.prepared_device_ordinal != device_ordinal) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_device_not_prepared");
  }
  std::unique_ptr<pih::CudaAllocator>* allocator_owner = nullptr;
  try {
    allocator_owner = &state.device_allocators[device_ordinal];
  } catch (...) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "cuda_allocator_tracking_failed");
  }
  auto& allocator = *allocator_owner;
  if (allocator == nullptr) {
    auto created = pih::CudaAllocator::Create(device_ordinal);
    if (!created.ok()) return FromStatus(created.status());
    allocator = std::move(*created);
  }
  auto created = allocator->allocate(bytes, alignment);
  if (!created.ok()) return FromStatus(created.status());
  pih_cuda_allocation_v1 completed{
      sizeof(completed), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1};
  ExportAllocation(*created, PIH_CUDA_MEMORY_DEVICE_V1, device_ordinal,
                   completed);
  bool inserted = completed.address == 0;
  try {
    inserted = completed.address == 0 ||
        state.allocations.emplace(
            completed.address,
            State::AllocationRecord{completed.bytes, completed.alignment,
                                    completed.generation, device_ordinal,
                                    completed.memory_kind}).second;
  } catch (...) {
    const auto cleanup = allocator->release(*created);
    if (!cleanup.ok()) state.resource_ownership_corrupted = true;
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "cuda_allocation_tracking_failed");
  }
  if (!inserted) {
    // A duplicate identity may alias the already tracked allocation. Freeing
    // it would invalidate the existing record, so quarantine the activation.
    state.resource_ownership_corrupted = true;
    return Status(PIH_STATUS_INTERNAL_V1,
                  "cuda_allocation_identity_duplicated");
  }
  *allocation = completed;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 DeallocateDevice(
    void* context, const pih_cuda_allocation_v1* allocation) {
  if (context == nullptr || allocation == nullptr ||
      allocation->struct_size != sizeof(*allocation) ||
      allocation->abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      allocation->device_ordinal < 0 ||
      allocation->memory_kind != PIH_CUDA_MEMORY_DEVICE_V1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_allocation_handle_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsCleanupAllowed(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_releasable");
  }
  if (allocation->address != 0) {
    const auto record = state.allocations.find(allocation->address);
    if (record == state.allocations.end() ||
        record->second.bytes != allocation->bytes ||
        record->second.alignment != allocation->alignment ||
        record->second.generation != allocation->generation ||
        record->second.device_ordinal != allocation->device_ordinal ||
        record->second.memory_kind != allocation->memory_kind) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "cuda_allocation_owner_mismatch");
    }
  }
  const auto found = state.device_allocators.find(allocation->device_ordinal);
  if (found == state.device_allocators.end()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_allocator_owner_missing");
  }
  auto device = pih::Device::Create(pih::DeviceType::kCuda,
                                    allocation->device_ordinal);
  if (!device.ok()) return FromStatus(device.status());
  const auto released = found->second->release(
      {reinterpret_cast<void*>(allocation->address), allocation->bytes,
       allocation->alignment, allocation->generation, *device});
  if (!released.ok()) return FromStatus(released);
  if (allocation->address != 0) {
    state.allocations.erase(allocation->address);
  }
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 AllocatePinnedHost(void* context, int32_t device_ordinal,
                                 uint64_t bytes, uint64_t alignment,
                                 pih_cuda_allocation_v1* allocation) {
  if (context == nullptr || device_ordinal != -1 || !ValidOutput(allocation)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "pinned_allocation_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (state.prepared_device_ordinal < 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_device_not_prepared");
  }
  if (state.pinned_allocator == nullptr) {
    auto created = pih::NvidiaPinnedHostAllocator::Create();
    if (!created.ok()) return FromStatus(created.status());
    state.pinned_allocator = std::move(*created);
  }
  auto created = state.pinned_allocator->allocate(bytes, alignment);
  if (!created.ok()) return FromStatus(created.status());
  pih_cuda_allocation_v1 completed{
      sizeof(completed), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1};
  ExportAllocation(*created, PIH_CUDA_MEMORY_PINNED_HOST_V1, -1,
                   completed);
  bool inserted = completed.address == 0;
  try {
    inserted = completed.address == 0 ||
        state.allocations.emplace(
            completed.address,
            State::AllocationRecord{completed.bytes, completed.alignment,
                                    completed.generation, -1,
                                    completed.memory_kind}).second;
  } catch (...) {
    const auto cleanup = created->data == nullptr
        ? cudaSuccess
        : cudaFreeHost(created->data);
    if (cleanup != cudaSuccess) state.resource_ownership_corrupted = true;
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "pinned_allocation_tracking_failed");
  }
  if (!inserted) {
    // Do not free an aliased address; the tracked allocation may own it.
    state.resource_ownership_corrupted = true;
    return Status(PIH_STATUS_INTERNAL_V1,
                  "pinned_allocation_identity_duplicated");
  }
  *allocation = completed;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 DeallocatePinnedHost(
    void* context, const pih_cuda_allocation_v1* allocation) {
  if (context == nullptr || allocation == nullptr ||
      allocation->struct_size != sizeof(*allocation) ||
      allocation->abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      allocation->device_ordinal != -1 ||
      allocation->memory_kind != PIH_CUDA_MEMORY_PINNED_HOST_V1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "pinned_allocation_handle_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsCleanupAllowed(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_releasable");
  }
  if (allocation->address != 0) {
    const auto record = state.allocations.find(allocation->address);
    if (record == state.allocations.end() ||
        record->second.bytes != allocation->bytes ||
        record->second.alignment != allocation->alignment ||
        record->second.generation != allocation->generation ||
        record->second.device_ordinal != allocation->device_ordinal ||
        record->second.memory_kind != allocation->memory_kind) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "pinned_allocation_owner_mismatch");
    }
  }
  if (state.pinned_allocator == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "pinned_allocator_owner_missing");
  }
  if (allocation->address != 0) {
    const auto released = cudaFreeHost(
        reinterpret_cast<void*>(allocation->address));
    if (released != cudaSuccess) {
      return FromStatus(pih::cuda_status(
          released, "cudaFreeHost capability"));
    }
    state.allocations.erase(allocation->address);
  }
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 CopyH2d(void* context, int32_t device_ordinal,
                      uintptr_t destination, const void* source,
                      uint64_t bytes) {
  if (context == nullptr || device_ordinal < 0 ||
      (bytes != 0 && (destination == 0 || source == nullptr))) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_copy_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (state.prepared_device_ordinal != device_ordinal) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_device_not_prepared");
  }
  if (bytes != 0 &&
      !OwnsDeviceRange(state, destination, bytes, device_ordinal)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_copy_destination_not_owned");
  }
  std::unique_ptr<pih::CudaMemoryCopier>* copier_owner = nullptr;
  try {
    copier_owner = &state.copiers[device_ordinal];
  } catch (...) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "cuda_copier_tracking_failed");
  }
  auto& copier = *copier_owner;
  if (copier == nullptr) {
    auto created = pih::CudaMemoryCopier::Create(device_ordinal);
    if (!created.ok()) return FromStatus(created.status());
    copier = std::move(*created);
  }
  auto device = pih::Device::Create(pih::DeviceType::kCuda, device_ordinal);
  if (!device.ok()) return FromStatus(device.status());
  return FromStatus(copier->copy(reinterpret_cast<void*>(destination), *device,
                                 source, pih::Device::Cpu(), bytes));
}

pih_status_v1 AllocateDeviceAbi(void* context, int32_t device_ordinal,
                                uint64_t bytes, uint64_t alignment,
                                pih_cuda_allocation_v1* allocation) noexcept {
  return ContainAbi(
      [&] {
        return AllocateDevice(context, device_ordinal, bytes, alignment,
                              allocation);
      },
      "cuda_device_allocation_failed");
}

pih_status_v1 DeallocateDeviceAbi(
    void* context, const pih_cuda_allocation_v1* allocation) noexcept {
  return ContainAbi([&] { return DeallocateDevice(context, allocation); },
                    "cuda_device_deallocation_failed");
}

pih_status_v1 AllocatePinnedHostAbi(
    void* context, int32_t device_ordinal, uint64_t bytes, uint64_t alignment,
    pih_cuda_allocation_v1* allocation) noexcept {
  return ContainAbi(
      [&] {
        return AllocatePinnedHost(context, device_ordinal, bytes, alignment,
                                  allocation);
      },
      "cuda_pinned_allocation_failed");
}

pih_status_v1 DeallocatePinnedHostAbi(
    void* context, const pih_cuda_allocation_v1* allocation) noexcept {
  return ContainAbi([&] { return DeallocatePinnedHost(context, allocation); },
                    "cuda_pinned_deallocation_failed");
}

pih_status_v1 CopyH2dAbi(void* context, int32_t device_ordinal,
                         uintptr_t destination, const void* source,
                         uint64_t bytes) noexcept {
  return ContainAbi(
      [&] {
        return CopyH2d(context, device_ordinal, destination, source, bytes);
      },
      "cuda_h2d_copy_failed");
}

pih_nvidia_cuda_memory_api_v1 memory_api{
    sizeof(pih_nvidia_cuda_memory_api_v1),
    PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1, &state,
    &AllocateDeviceAbi, &DeallocateDeviceAbi, &AllocatePinnedHostAbi,
    &DeallocatePinnedHostAbi, &CopyH2dAbi};

pih_status_v1 RetainPrimaryContext(void* context, int32_t device_ordinal,
                                   uint32_t context_flags,
                                   uintptr_t* retained_context) {
  if (context == nullptr || device_ordinal < 0 || retained_context == nullptr ||
      *retained_context != 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_context_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (state.prepared_device_ordinal != device_ordinal) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_device_not_prepared");
  }
  auto created = state.runtime_driver.retain_primary_context(
      device_ordinal, context_flags);
  if (!created.ok()) return FromStatus(created.status());
  if (*created == 0) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "cuda_context_identity_invalid");
  }
  try {
    auto [record, inserted] = state.contexts.emplace(
        *created, std::pair<int32_t, uint32_t>{device_ordinal, 1});
    if (!inserted) {
      if (record->second.first != device_ordinal ||
          record->second.second == UINT32_MAX) {
        if (!RollbackPrimaryContext(device_ordinal, *created)) {
          state.resource_ownership_corrupted = true;
        }
        return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                      "cuda_context_owner_conflict");
      }
      ++record->second.second;
    }
  } catch (...) {
    if (!RollbackPrimaryContext(device_ordinal, *created)) {
      state.resource_ownership_corrupted = true;
    }
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "cuda_context_tracking_failed");
  }
  *retained_context = *created;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 BindRuntime(void* context, int32_t device_ordinal,
                          uintptr_t retained_context) {
  if (context == nullptr || device_ordinal < 0 || retained_context == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_runtime_binding_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  const auto found = state.contexts.find(retained_context);
  if (found == state.contexts.end() || found->second.first != device_ordinal) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_owner_mismatch");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  return FromStatus(state.runtime_driver.bind_runtime(device_ordinal,
                                                       retained_context));
}

pih_status_v1 CreateStream(void* context, uintptr_t retained_context,
                           uintptr_t* resource) {
  if (context == nullptr || retained_context == 0 || resource == nullptr ||
      *resource != 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_stream_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (!state.contexts.contains(retained_context)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_owner_missing");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  auto created = state.runtime_driver.create_nonblocking_stream(
      retained_context);
  if (!created.ok()) return FromStatus(created.status());
  if (*created == 0) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "cuda_stream_identity_invalid");
  }
  bool inserted = false;
  try {
    inserted = state.streams.emplace(*created, retained_context).second;
  } catch (...) {
    const auto cleanup =
        cuStreamDestroy(reinterpret_cast<CUstream>(*created));
    if (cleanup != CUDA_SUCCESS) state.resource_ownership_corrupted = true;
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "cuda_stream_tracking_failed");
  }
  if (!inserted) {
    state.resource_ownership_corrupted = true;
    return Status(PIH_STATUS_INTERNAL_V1, "cuda_stream_identity_duplicated");
  }
  *resource = *created;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 CreateEvent(void* context, uintptr_t retained_context,
                          uintptr_t* resource) {
  if (context == nullptr || retained_context == 0 || resource == nullptr ||
      *resource != 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_event_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (!state.contexts.contains(retained_context)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_owner_missing");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  auto created = state.runtime_driver.create_disable_timing_event(
      retained_context);
  if (!created.ok()) return FromStatus(created.status());
  if (*created == 0) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "cuda_event_identity_invalid");
  }
  bool inserted = false;
  try {
    inserted = state.events.emplace(*created, retained_context).second;
  } catch (...) {
    const auto cleanup = cuEventDestroy(reinterpret_cast<CUevent>(*created));
    if (cleanup != CUDA_SUCCESS) state.resource_ownership_corrupted = true;
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "cuda_event_tracking_failed");
  }
  if (!inserted) {
    state.resource_ownership_corrupted = true;
    return Status(PIH_STATUS_INTERNAL_V1, "cuda_event_identity_duplicated");
  }
  *resource = *created;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 DestroyEvent(void* context, uintptr_t resource) {
  if (context == nullptr || resource == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_event_handle_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsCleanupAllowed(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_releasable");
  }
  const auto found = state.events.find(resource);
  if (found == state.events.end()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_event_owner_mismatch");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(found->second)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  const auto destroyed = cuEventDestroy(reinterpret_cast<CUevent>(resource));
  if (destroyed != CUDA_SUCCESS) {
    return FromStatus(pih::cuda_driver_status(
        destroyed, "cuEventDestroy capability"));
  }
  state.events.erase(found);
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 DestroyStream(void* context, uintptr_t resource) {
  if (context == nullptr || resource == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_stream_handle_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsCleanupAllowed(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_releasable");
  }
  const auto found = state.streams.find(resource);
  if (found == state.streams.end()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_stream_owner_mismatch");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(found->second)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  const auto destroyed = cuStreamDestroy(reinterpret_cast<CUstream>(resource));
  if (destroyed != CUDA_SUCCESS) {
    return FromStatus(pih::cuda_driver_status(
        destroyed, "cuStreamDestroy capability"));
  }
  state.streams.erase(found);
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ReleasePrimaryContext(void* context, int32_t device_ordinal,
                                    uintptr_t retained_context) {
  if (context == nullptr || device_ordinal < 0 || retained_context == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_context_handle_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsCleanupAllowed(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_releasable");
  }
  const auto found = state.contexts.find(retained_context);
  if (found == state.contexts.end() || found->second.first != device_ordinal) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_owner_mismatch");
  }
  for (const auto& [_, owner] : state.streams) {
    if (owner == retained_context) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "cuda_context_has_live_streams");
    }
  }
  for (const auto& [_, owner] : state.events) {
    if (owner == retained_context) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "cuda_context_has_live_events");
    }
  }
  CUdevice device{};
  auto released = cuDeviceGet(&device, device_ordinal);
  if (released != CUDA_SUCCESS) {
    return FromStatus(pih::cuda_driver_status(
        released, "cuDeviceGet primary context release capability"));
  }
  CUcontext current = nullptr;
  released = cuCtxGetCurrent(&current);
  if (released != CUDA_SUCCESS) {
    return FromStatus(pih::cuda_driver_status(
        released, "cuCtxGetCurrent primary context release capability"));
  }
  if (reinterpret_cast<uintptr_t>(current) == retained_context) {
    released = cuCtxSetCurrent(nullptr);
    if (released != CUDA_SUCCESS) {
      return FromStatus(pih::cuda_driver_status(
          released, "cuCtxSetCurrent primary context release capability"));
    }
  }
  released = cuDevicePrimaryCtxRelease(device);
  if (released != CUDA_SUCCESS) {
    return FromStatus(pih::cuda_driver_status(
        released, "cuDevicePrimaryCtxRelease capability"));
  }
  if (--found->second.second == 0) state.contexts.erase(found);
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 RetainPrimaryContextAbi(
    void* context, int32_t device_ordinal, uint32_t context_flags,
    uintptr_t* retained_context) noexcept {
  return ContainAbi(
      [&] {
        return RetainPrimaryContext(context, device_ordinal, context_flags,
                                    retained_context);
      },
      "cuda_context_retain_failed");
}

pih_status_v1 BindRuntimeAbi(void* context, int32_t device_ordinal,
                             uintptr_t retained_context) noexcept {
  return ContainAbi(
      [&] { return BindRuntime(context, device_ordinal, retained_context); },
      "cuda_runtime_binding_failed");
}

pih_status_v1 CreateStreamAbi(void* context, uintptr_t retained_context,
                              uintptr_t* resource) noexcept {
  return ContainAbi(
      [&] { return CreateStream(context, retained_context, resource); },
      "cuda_stream_creation_failed");
}

pih_status_v1 CreateEventAbi(void* context, uintptr_t retained_context,
                             uintptr_t* resource) noexcept {
  return ContainAbi(
      [&] { return CreateEvent(context, retained_context, resource); },
      "cuda_event_creation_failed");
}

pih_status_v1 DestroyEventAbi(void* context, uintptr_t resource) noexcept {
  return ContainAbi([&] { return DestroyEvent(context, resource); },
                    "cuda_event_destruction_failed");
}

pih_status_v1 DestroyStreamAbi(void* context, uintptr_t resource) noexcept {
  return ContainAbi([&] { return DestroyStream(context, resource); },
                    "cuda_stream_destruction_failed");
}

pih_status_v1 ReleasePrimaryContextAbi(
    void* context, int32_t device_ordinal,
    uintptr_t retained_context) noexcept {
  return ContainAbi(
      [&] {
        return ReleasePrimaryContext(context, device_ordinal,
                                     retained_context);
      },
      "cuda_context_release_failed");
}

pih_nvidia_cuda_resources_api_v1 resources_api{
    sizeof(pih_nvidia_cuda_resources_api_v1),
    PIH_NVIDIA_CUDA_RESOURCES_ABI_VERSION_V1, &state,
    &RetainPrimaryContextAbi, &BindRuntimeAbi, &CreateStreamAbi,
    &CreateEventAbi, &DestroyEventAbi, &DestroyStreamAbi,
    &ReleasePrimaryContextAbi};

pih_status_v1 ActivateContext(void* context, uintptr_t retained_context) {
  if (context == nullptr || retained_context == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_context_activate_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (!state.contexts.contains(retained_context)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_owner_mismatch");
  }
  return FromStatus(pih::cuda_driver_status(
      cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)),
      "cuCtxSetCurrent capability"));
}

pih_status_v1 ValidatePinnedHost(void* context, uintptr_t retained_context,
                                 uintptr_t pointer) {
  if (context == nullptr || retained_context == 0 || pointer == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_pinned_host_validation_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (!state.contexts.contains(retained_context)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_owner_mismatch");
  }
  if (!OwnsPinnedRange(state, pointer, 1)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_pinned_host_allocation_not_owned");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  cudaPointerAttributes attributes{};
  const auto result = cudaPointerGetAttributes(
      &attributes, reinterpret_cast<const void*>(pointer));
  if (result != cudaSuccess) {
    return FromStatus(pih::cuda_status(
        result, "cudaPointerGetAttributes capability"));
  }
  return attributes.type == cudaMemoryTypeHost
      ? Status(PIH_STATUS_OK_V1)
      : Status(PIH_STATUS_FAILED_PRECONDITION_V1,
               "cuda_pointer_is_not_pinned_host");
}

pih_status_v1 CopyAsync(void* context, uintptr_t retained_context,
                        uintptr_t destination, uintptr_t source,
                        uint64_t bytes, uint32_t copy_kind,
                        uintptr_t stream) {
  if (context == nullptr || retained_context == 0 || destination == 0 ||
      source == 0 || bytes == 0 || stream == 0 ||
      (copy_kind != PIH_CUDA_COPY_H2D_V1 &&
       copy_kind != PIH_CUDA_COPY_D2H_V1 &&
       copy_kind != PIH_CUDA_COPY_D2D_V1)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "cuda_async_copy_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  const auto context_owner = state.contexts.find(retained_context);
  if (context_owner == state.contexts.end() ||
      !state.streams.contains(stream) ||
      state.streams.at(stream) != retained_context) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_async_copy_owner_mismatch");
  }
  const auto device_ordinal = context_owner->second.first;
  const bool destination_owned =
      copy_kind == PIH_CUDA_COPY_D2H_V1 ||
      OwnsDeviceRange(state, destination, bytes, device_ordinal);
  const bool source_owned =
      copy_kind == PIH_CUDA_COPY_H2D_V1 ||
      OwnsDeviceRange(state, source, bytes, device_ordinal);
  const auto host_address = copy_kind == PIH_CUDA_COPY_H2D_V1
      ? source
      : destination;
  const bool host_owned = copy_kind == PIH_CUDA_COPY_D2D_V1 ||
                          OwnsPinnedRange(state, host_address, bytes);
  if (!destination_owned || !source_owned || !host_owned) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_async_copy_allocation_not_owned");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  if (copy_kind != PIH_CUDA_COPY_D2D_V1) {
    cudaPointerAttributes attributes{};
    const auto attributes_status = cudaPointerGetAttributes(
        &attributes, reinterpret_cast<const void*>(host_address));
    if (attributes_status != cudaSuccess) {
      return FromStatus(pih::cuda_status(
          attributes_status, "cudaPointerGetAttributes async copy"));
    }
    if (attributes.type != cudaMemoryTypeHost) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "cuda_async_copy_host_memory_not_pinned");
    }
  }
  const auto kind = copy_kind == PIH_CUDA_COPY_H2D_V1
      ? cudaMemcpyHostToDevice
      : copy_kind == PIH_CUDA_COPY_D2H_V1 ? cudaMemcpyDeviceToHost
                                         : cudaMemcpyDeviceToDevice;
  return FromStatus(pih::cuda_status(
      cudaMemcpyAsync(reinterpret_cast<void*>(destination),
                      reinterpret_cast<const void*>(source), bytes, kind,
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemcpyAsync capability"));
}

pih_status_v1 MemsetAsync(void* context, uintptr_t retained_context,
                          uintptr_t destination, uint32_t byte_value,
                          uint64_t bytes, uintptr_t stream) {
  if (context == nullptr || retained_context == 0 || destination == 0 ||
      byte_value > 255 || bytes == 0 || stream == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "cuda_async_memset_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  const auto context_owner = state.contexts.find(retained_context);
  if (context_owner == state.contexts.end() ||
      !state.streams.contains(stream) ||
      state.streams.at(stream) != retained_context) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_async_memset_owner_mismatch");
  }
  if (!OwnsDeviceRange(state, destination, bytes,
                       context_owner->second.first)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_async_memset_allocation_not_owned");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  return FromStatus(pih::cuda_status(
      cudaMemsetAsync(reinterpret_cast<void*>(destination),
                      static_cast<int>(byte_value), bytes,
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemsetAsync capability"));
}

pih_status_v1 RecordEvent(void* context, uintptr_t retained_context,
                          uintptr_t event, uintptr_t stream) {
  if (context == nullptr || retained_context == 0 || event == 0 ||
      stream == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "cuda_event_record_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (!state.contexts.contains(retained_context) ||
      !state.events.contains(event) ||
      state.events.at(event) != retained_context ||
      !state.streams.contains(stream) ||
      state.streams.at(stream) != retained_context) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_event_owner_mismatch");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  return FromStatus(pih::cuda_status(
      cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaEventRecord capability"));
}

pih_status_v1 QueryEvent(void* context, uintptr_t retained_context,
                         uintptr_t event, uint32_t* event_status) {
  if (event_status != nullptr) *event_status = 0;
  if (context == nullptr || retained_context == 0 || event == 0 ||
      event_status == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "cuda_event_query_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (!state.contexts.contains(retained_context) ||
      !state.events.contains(event) ||
      state.events.at(event) != retained_context) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_event_owner_mismatch");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  const auto result = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
  if (result == cudaSuccess) {
    *event_status = PIH_CUDA_EVENT_COMPLETE_V1;
    return Status(PIH_STATUS_OK_V1);
  }
  if (result == cudaErrorNotReady) {
    *event_status = PIH_CUDA_EVENT_PENDING_V1;
    return Status(PIH_STATUS_OK_V1);
  }
  return FromStatus(pih::cuda_status(result, "cudaEventQuery capability"));
}

pih_status_v1 SynchronizeStream(void* context, uintptr_t retained_context,
                                uintptr_t stream) {
  if (context == nullptr || retained_context == 0 || stream == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_stream_synchronize_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (!state.contexts.contains(retained_context) ||
      !state.streams.contains(stream) ||
      state.streams.at(stream) != retained_context) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_stream_synchronize_owner_mismatch");
  }
  if (cuCtxSetCurrent(reinterpret_cast<CUcontext>(retained_context)) !=
      CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_context_bind_failed");
  }
  return FromStatus(pih::cuda_status(
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
      "cudaStreamSynchronize capability"));
}

pih_status_v1 ActivateContextAbi(void* context,
                                 uintptr_t retained_context) noexcept {
  return ContainAbi(
      [&] { return ActivateContext(context, retained_context); },
      "cuda_context_activation_failed");
}

pih_status_v1 ValidatePinnedHostAbi(void* context,
                                    uintptr_t retained_context,
                                    uintptr_t pointer) noexcept {
  return ContainAbi(
      [&] { return ValidatePinnedHost(context, retained_context, pointer); },
      "cuda_pinned_host_validation_failed");
}

pih_status_v1 CopyAsyncAbi(
    void* context, uintptr_t retained_context, uintptr_t destination,
    uintptr_t source, uint64_t bytes, uint32_t copy_kind,
    uintptr_t stream) noexcept {
  return ContainAbi(
      [&] {
        return CopyAsync(context, retained_context, destination, source, bytes,
                         copy_kind, stream);
      },
      "cuda_async_copy_failed");
}

pih_status_v1 MemsetAsyncAbi(
    void* context, uintptr_t retained_context, uintptr_t destination,
    uint32_t byte_value, uint64_t bytes, uintptr_t stream) noexcept {
  return ContainAbi(
      [&] {
        return MemsetAsync(context, retained_context, destination, byte_value,
                           bytes, stream);
      },
      "cuda_async_memset_failed");
}

pih_status_v1 RecordEventAbi(void* context, uintptr_t retained_context,
                             uintptr_t event, uintptr_t stream) noexcept {
  return ContainAbi(
      [&] { return RecordEvent(context, retained_context, event, stream); },
      "cuda_event_record_failed");
}

pih_status_v1 QueryEventAbi(void* context, uintptr_t retained_context,
                            uintptr_t event,
                            uint32_t* event_status) noexcept {
  return ContainAbi(
      [&] {
        return QueryEvent(context, retained_context, event, event_status);
      },
      "cuda_event_query_failed");
}

pih_status_v1 SynchronizeStreamAbi(void* context,
                                   uintptr_t retained_context,
                                   uintptr_t stream) noexcept {
  return ContainAbi(
      [&] { return SynchronizeStream(context, retained_context, stream); },
      "cuda_stream_synchronize_failed");
}

pih_status_v1 RequireCleanLastErrorAbi(void* context,
                                      uintptr_t retained_context) noexcept {
  return ContainAbi([&] {
    if (context != &state || retained_context == 0)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "cuda_last_error_context_invalid");
    std::lock_guard lock(state.memory_mutex);
    if (!IsReady(state) || !state.contexts.contains(retained_context))
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "cuda_last_error_owner_invalid");
    CUcontext current = nullptr;
    const auto queried = cuCtxGetCurrent(&current);
    if (queried != CUDA_SUCCESS)
      return FromStatus(pih::cuda_driver_status(queried, "cuCtxGetCurrent last-error capability"));
    if (reinterpret_cast<uintptr_t>(current) != retained_context)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "cuda_last_error_current_context_mismatch");
    return FromStatus(pih::cuda_status(cudaPeekAtLastError(), "cudaPeekAtLastError capability"));
  }, "cuda_last_error_probe_failed");
}

pih_nvidia_cuda_async_api_v1 async_api{
    sizeof(pih_nvidia_cuda_async_api_v1),
    PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1, &state, &ActivateContextAbi,
    &ValidatePinnedHostAbi, &CopyAsyncAbi, &MemsetAsyncAbi, &RecordEventAbi,
    &QueryEventAbi, &SynchronizeStreamAbi, &RequireCleanLastErrorAbi};

pih_status_v1 PrepareDevice(void* context, int32_t device_ordinal,
                            uint32_t expected_sm_major,
                            uint32_t expected_sm_minor) {
  if (context != &state || device_ordinal < 0 || expected_sm_major == 0 ||
      expected_sm_minor > 9) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "cuda_device_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (!IsReady(state)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_provider_not_ready");
  }
  if (state.prepared_device_ordinal >= 0) {
    if (state.prepared_device_ordinal != device_ordinal ||
        state.prepared_sm_major != expected_sm_major ||
        state.prepared_sm_minor != expected_sm_minor) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "cuda_device_already_prepared");
    }
    return Status(PIH_STATUS_OK_V1);
  }
  if (cuInit(0) != CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "cuda_init_failed");
  }
  CUdevice device{};
  if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_device_unavailable");
  }
  int sm_major = 0;
  int sm_minor = 0;
  if (cuDeviceGetAttribute(&sm_major,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           device) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&sm_minor,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           device) != CUDA_SUCCESS) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_device_identity_failed");
  }
  if (sm_major != static_cast<int>(expected_sm_major) ||
      sm_minor != static_cast<int>(expected_sm_minor)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_compute_capability_mismatch");
  }
  state.prepared_device_ordinal = device_ordinal;
  state.prepared_sm_major = expected_sm_major;
  state.prepared_sm_minor = expected_sm_minor;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 PrepareDeviceAbi(void* context, int32_t device_ordinal,
                               uint32_t expected_sm_major,
                               uint32_t expected_sm_minor) noexcept {
  return ContainAbi(
      [&] {
        return PrepareDevice(context, device_ordinal, expected_sm_major,
                             expected_sm_minor);
      },
      "cuda_device_prepare_failed");
}

pih_nvidia_cuda_api_v1 cuda_api{sizeof(pih_nvidia_cuda_api_v1),
                                PIH_NVIDIA_CUDA_ABI_VERSION_V1, &state,
                                &PrepareDeviceAbi};

pih_status_v1 Register(void* context) {
  auto& state = *static_cast<State*>(context);
  if (state.phase != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  if (state.host->register_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_registry_missing");
  }
  pih_capability_v1 capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = PIH_CAPABILITY_ABI_VERSION_V1;
  capability.threading_model = PIH_CAPABILITY_THREADING_SERIALIZED_V1;
  capability.scope = PIH_CAPABILITY_SCOPE_PROCESS_V1;
  capability.cardinality = PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
  capability.capability_id = "device.cuda-runtime.v1";
  capability.contract_id = "pih.device.cuda-runtime.v1";
  capability.api = &cuda_api;
  auto status = CheckedCapabilityStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  capability.capability_id = "device.cuda-memory.v1";
  capability.contract_id = "pih.device.cuda-memory.v1";
  capability.api = &memory_api;
  status = CheckedCapabilityStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  capability.capability_id = "device.cuda-resources.v1";
  capability.contract_id = "pih.device.cuda-resources.v1";
  capability.api = &resources_api;
  status = CheckedCapabilityStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  capability.capability_id = "device.cuda-async.v1";
  capability.contract_id = "pih.device.cuda-async.v1";
  capability.api = &async_api;
  status = CheckedCapabilityStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  return Advance(context, 0);
}

pih_status_v1 Configure(void* context) {
  auto& state = *static_cast<State*>(context);
  if (state.phase != 1) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  if (state.host->resolve_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_resolver_missing");
  }
  const void* api = nullptr;
  const auto resolved = CheckedCapabilityStatus(
      state.host->resolve_capability(
          state.host->context, "platform.linux.v1", "pih.platform.linux.v1",
          PIH_CAPABILITY_SCOPE_PROCESS_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api),
      "capability_resolver_status_invalid");
  if (!pih_status_is_ok_v1(&resolved)) return resolved;
  const auto* platform = static_cast<const pih_platform_linux_api_v1*>(api);
  if (platform == nullptr || platform->struct_size != sizeof(*platform) ||
      platform->contract_version != PIH_PLATFORM_LINUX_ABI_VERSION_V1 ||
      platform->context == nullptr || platform->validate_runtime == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "linux_platform_capability_invalid");
  }
  const auto advanced = Advance(context, 1);
  if (!pih_status_is_ok_v1(&advanced)) return advanced;
  state.platform = platform;
  return advanced;
}
pih_status_v1 Start(void* context) {
  if (context != &state || state.phase != 2 || state.platform == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_start_configuration_invalid");
  }
  // Registration exposes tables before providers are configured. Defer calls
  // into the platform until the stack-wide configure barrier has completed;
  // a Lock need not put platform before backend in its plugin list.
  const auto* platform = state.platform;
  const auto platform_status = CheckedCapabilityStatus(
      platform->validate_runtime(platform->context),
      "platform_provider_status_invalid");
  if (!pih_status_is_ok_v1(&platform_status)) return platform_status;
  return Advance(context, 2);
}
pih_status_v1 Ready(void* context) { return Advance(context, 3); }
pih_status_v1 Drain(void* context) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (state.resource_ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_resource_ownership_corrupted");
  }
  if (!state.allocations.empty() || !state.events.empty() ||
      !state.streams.empty() || !state.contexts.empty()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_resources_still_live");
  }
  return Advance(context, 4);
}
pih_status_v1 Stop(void* context) {
  if (context == nullptr) return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                                        "lifecycle_context_invalid");
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.memory_mutex);
  if (state.phase == 3) {
    state.phase = 6;
    return Status(PIH_STATUS_OK_V1);
  }
  return Advance(context, 5);
}
pih_status_v1 Dispose(void* context) {
  auto& state = *static_cast<State*>(context);
  {
    std::lock_guard lock(state.memory_mutex);
    if (state.phase != 1 && state.phase != 2 && state.phase != 6) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "lifecycle_order_invalid");
    }
    if (state.resource_ownership_corrupted) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "cuda_resource_ownership_corrupted");
    }
    if (!state.allocations.empty() || !state.events.empty() ||
        !state.streams.empty() || !state.contexts.empty()) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "cuda_resources_still_live");
    }
    state.copiers.clear();
    state.device_allocators.clear();
    state.pinned_allocator.reset();
    state.prepared_device_ordinal = -1;
    state.prepared_sm_major = 0;
    state.prepared_sm_minor = 0;
    state.resource_ownership_corrupted = false;
    state.platform = nullptr;
    state.host = nullptr;
    state.phase = 7;
  }
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 RegisterAbi(void* context) noexcept {
  return ContainAbi([&] { return Register(context); },
                    "cuda_registration_failed");
}

pih_status_v1 ConfigureAbi(void* context) noexcept {
  return ContainAbi([&] { return Configure(context); },
                    "cuda_configuration_failed");
}

pih_status_v1 StartAbi(void* context) noexcept {
  return ContainAbi([&] { return Start(context); }, "cuda_start_failed");
}

pih_status_v1 ReadyAbi(void* context) noexcept {
  return ContainAbi([&] { return Ready(context); }, "cuda_ready_failed");
}

pih_status_v1 DrainAbi(void* context) noexcept {
  return ContainAbi([&] { return Drain(context); }, "cuda_drain_failed");
}

pih_status_v1 StopAbi(void* context) noexcept {
  return ContainAbi([&] { return Stop(context); }, "cuda_stop_failed");
}

pih_status_v1 DisposeAbi(void* context) noexcept {
  return ContainAbi([&] { return Dispose(context); },
                    "cuda_dispose_failed");
}

}  // namespace

extern "C" PIH_PLUGIN_EXPORT pih_status_v1
pih_plugin_entry_v1(const pih_host_api_v1* host,
                    pih_plugin_api_v1* plugin) noexcept {
  if (!pih_host_api_is_valid_v1(host)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "host_api_invalid");
  }
  if (!pih_plugin_api_accepts_v1(plugin)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "plugin_api_invalid");
  }
  if (state.host != nullptr || state.phase != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "plugin_entry_already_bound");
  }
  state.host = host;
  plugin->abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  plugin->plugin_id = "pih.backend.nvidia-cuda";
  plugin->plugin_version = "1.0.0";
  plugin->context = &state;
  plugin->lifecycle.struct_size = sizeof(pih_plugin_lifecycle_v1);
  plugin->lifecycle.abi_version = PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1;
  plugin->lifecycle.register_plugin = &RegisterAbi;
  plugin->lifecycle.configure = &ConfigureAbi;
  plugin->lifecycle.start = &StartAbi;
  plugin->lifecycle.ready = &ReadyAbi;
  plugin->lifecycle.drain = &DrainAbi;
  plugin->lifecycle.stop = &StopAbi;
  plugin->lifecycle.dispose = &DisposeAbi;
  return Status(PIH_STATUS_OK_V1);
}
