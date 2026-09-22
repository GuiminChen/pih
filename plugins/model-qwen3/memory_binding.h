#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/contracts/nvidia_cuda_memory_v1.h"
#include "pih/core/memory_copier.h"

namespace pih::qwen_plugin {

inline bool ValidMemoryApi(const pih_nvidia_cuda_memory_api_v1& api) noexcept {
  return api.struct_size == sizeof(api) &&
      api.contract_version == PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 &&
      api.context && api.allocate_device && api.deallocate_device &&
      api.allocate_pinned_host && api.deallocate_pinned_host && api.copy_h2d;
}

inline Status MemoryStatus(const pih_status_v1& status) {
  if (!pih_status_is_valid_v1(&status))
    return Status::Internal("Qwen memory provider status ABI invalid");
  const std::string message(status.message,
      std::find(status.message, status.message + sizeof(status.message), '\0'));
  switch (status.code) {
    case PIH_STATUS_OK_V1: return Status::Ok();
    case PIH_STATUS_INVALID_ARGUMENT_V1: return Status::InvalidArgument(message);
    case PIH_STATUS_FAILED_PRECONDITION_V1: return Status::FailedPrecondition(message);
    case PIH_STATUS_RESOURCE_EXHAUSTED_V1: return Status::ResourceExhausted(message);
    case PIH_STATUS_UNAVAILABLE_V1: return Status::Unavailable(message);
    case PIH_STATUS_DEADLINE_EXCEEDED_V1: return Status::DeadlineExceeded(message);
    default: return Status::Internal(message);
  }
}

// Provider and context must outlive every allocation. The Allocator destruction
// interface cannot propagate an error: failed retirement fail-stops the worker,
// rather than unloading the provider while ownership remains uncertain.
template<bool Pinned>
class CapabilityAllocator final
    : public std::conditional_t<Pinned, RegisteredPinnedAllocator, Allocator> {
 public:
  CapabilityAllocator(const pih_nvidia_cuda_memory_api_v1& api, std::int32_t ordinal)
      : api_(&api), ordinal_(Pinned ? -1 : ordinal) {}

  Result<Allocation> allocate(std::uint64_t bytes, std::uint64_t alignment) override {
    if (!ValidMemoryApi(*api_) || (!Pinned && ordinal_ < 0))
      return Status::FailedPrecondition("Qwen memory provider binding invalid");
    if (!alignment || (alignment & (alignment - 1)) || alignment > 256 ||
        bytes > std::numeric_limits<std::size_t>::max())
      return Status::InvalidArgument("Qwen memory allocation shape invalid");
    pih_cuda_allocation_v1 value{};
    value.struct_size = sizeof(value);
    value.abi_version = PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1;
    const auto allocate = Pinned ? api_->allocate_pinned_host : api_->allocate_device;
    auto status = MemoryStatus(allocate(api_->context, ordinal_, bytes, alignment, &value));
    if (!status.ok()) return status;
    if (value.struct_size != sizeof(value) ||
        value.abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
        value.memory_kind != Kind() || value.device_ordinal != ordinal_ ||
        value.bytes != bytes || !value.generation || value.alignment < alignment ||
        !value.alignment || (value.alignment & (value.alignment - 1)) ||
        (bytes == 0 ? value.address != 0 :
         (!value.address || value.address % value.alignment ||
          bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - value.address))) {
      Retire(value);
      return Status::FailedPrecondition("Qwen memory provider returned a foreign allocation");
    }
    const auto device = Pinned ? Device::Cpu() : Device::Create(DeviceType::kCuda, ordinal_).value();
    return Allocation{reinterpret_cast<void*>(value.address), value.bytes,
                      value.alignment, value.generation, device};
  }

  void deallocate(Allocation allocation) noexcept override {
    const auto expected = Pinned ? Device::Cpu() : Device::Create(DeviceType::kCuda, ordinal_).value();
    if (allocation.device != expected) std::abort();
    Retire({sizeof(pih_cuda_allocation_v1), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1,
            reinterpret_cast<std::uintptr_t>(allocation.data), allocation.bytes,
            allocation.alignment, allocation.generation, ordinal_, Kind()});
  }

 private:
  static constexpr std::uint32_t Kind() noexcept {
    return Pinned ? PIH_CUDA_MEMORY_PINNED_HOST_V1 : PIH_CUDA_MEMORY_DEVICE_V1;
  }
  void Retire(const pih_cuda_allocation_v1& value) noexcept {
    try {
      const auto release = Pinned ? api_->deallocate_pinned_host : api_->deallocate_device;
      const auto status = release(api_->context, &value);
      if (!pih_status_is_ok_v1(&status)) std::abort();
    } catch (...) { std::abort(); }
  }
  const pih_nvidia_cuda_memory_api_v1* api_;
  std::int32_t ordinal_;
};

using CapabilityDeviceAllocator = CapabilityAllocator<false>;
using CapabilityPinnedAllocator = CapabilityAllocator<true>;

class CapabilityMemoryCopier final : public MemoryCopier {
 public:
  CapabilityMemoryCopier(const pih_nvidia_cuda_memory_api_v1& api, std::int32_t ordinal)
      : api_(&api), ordinal_(ordinal) {}
  Status copy(void* destination, Device destination_device, const void* source,
              Device source_device, std::uint64_t bytes) override {
    if (!ValidMemoryApi(*api_) || ordinal_ < 0)
      return Status::FailedPrecondition("Qwen copy provider binding invalid");
    if (source_device != Device::Cpu() || destination_device.type() != DeviceType::kCuda ||
        destination_device.index() != ordinal_ || (bytes && (!destination || !source)))
      return Status::InvalidArgument("Qwen copy requires CPU to exact-device memory");
    return MemoryStatus(api_->copy_h2d(api_->context, ordinal_,
                                      reinterpret_cast<std::uintptr_t>(destination), source, bytes));
  }
 private:
  const pih_nvidia_cuda_memory_api_v1* api_;
  std::int32_t ordinal_;
};
}  // namespace pih::qwen_plugin
