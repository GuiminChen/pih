#include <gtest/gtest.h>

#include "../../../plugins/model-qwen3/memory_binding.h"
#include "../../../plugins/model-qwen3/execution_binding.h"

namespace {
using namespace pih;
using namespace pih::qwen_plugin;
struct Provider {
  unsigned allocated{}, released{}, copies{};
  bool foreign{}, fail_release{};
  pih_cuda_allocation_v1 last{};
};
pih_status_v1 Allocate(void* context, int32_t ordinal, uint64_t bytes,
                       uint64_t, pih_cuda_allocation_v1* value) {
  auto& state = *static_cast<Provider*>(context);
  ++state.allocated;
  *value = {sizeof(*value), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1,
            bytes ? uintptr_t(0x10000) : 0, bytes, 256, 7,
            state.foreign ? ordinal + 1 : ordinal,
            ordinal == -1 ? uint32_t(PIH_CUDA_MEMORY_PINNED_HOST_V1) : uint32_t(PIH_CUDA_MEMORY_DEVICE_V1)};
  return ExecutionBindingStatus(PIH_STATUS_OK_V1, "");
}
pih_status_v1 Release(void* context, const pih_cuda_allocation_v1* value) {
  auto& state = *static_cast<Provider*>(context);
  ++state.released;
  state.last = *value;
  return ExecutionBindingStatus(state.fail_release ? PIH_STATUS_INTERNAL_V1 : PIH_STATUS_OK_V1, "");
}
pih_status_v1 Copy(void* context, int32_t ordinal, uintptr_t destination,
                   const void* source, uint64_t bytes) {
  ++static_cast<Provider*>(context)->copies;
  EXPECT_EQ(ordinal, 0);
  EXPECT_EQ(destination, 0x10000U);
  EXPECT_NE(source, nullptr);
  EXPECT_EQ(bytes, 32U);
  return ExecutionBindingStatus(PIH_STATUS_OK_V1, "");
}
pih_nvidia_cuda_memory_api_v1 Api(Provider& provider) {
  return {sizeof(pih_nvidia_cuda_memory_api_v1), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1,
          &provider, Allocate, Release, Allocate, Release, Copy};
}
TEST(NativeQwenMemoryBinding, DeviceAllocationRetiresExactProviderIdentity) {
  Provider provider;
  auto api = Api(provider);
  CapabilityDeviceAllocator allocator(api, 0);
  auto value = allocator.allocate(32, 128);
  ASSERT_TRUE(value.ok());
  EXPECT_EQ(value->device, Device::Create(DeviceType::kCuda, 0).value());
  EXPECT_EQ(value->generation, 7U);
  allocator.deallocate(*value);
  EXPECT_EQ(provider.released, 1U);
  EXPECT_EQ(provider.last.address, 0x10000U);
  EXPECT_EQ(provider.last.bytes, 32U);
  EXPECT_EQ(provider.last.alignment, 256U);
  EXPECT_EQ(provider.last.generation, 7U);
  EXPECT_EQ(provider.last.memory_kind, PIH_CUDA_MEMORY_DEVICE_V1);
}
TEST(NativeQwenMemoryBinding, PinnedAllocationUsesCpuIdentityAndMinusOneOrdinal) {
  Provider provider;
  auto api = Api(provider);
  CapabilityPinnedAllocator allocator(api, -1);
  auto value = allocator.allocate(32, 256);
  ASSERT_TRUE(value.ok());
  EXPECT_EQ(value->device, Device::Cpu());
  allocator.deallocate(*value);
  EXPECT_EQ(provider.last.device_ordinal, -1);
  EXPECT_EQ(provider.last.memory_kind, PIH_CUDA_MEMORY_PINNED_HOST_V1);
}
TEST(NativeQwenMemoryBinding, RejectsBadApiAndAlignmentWithoutAllocation) {
  Provider provider;
  auto api = Api(provider);
  CapabilityDeviceAllocator allocator(api, 0);
  EXPECT_FALSE(allocator.allocate(32, 3).ok());
  api.copy_h2d = nullptr;
  EXPECT_FALSE(allocator.allocate(32, 128).ok());
  EXPECT_EQ(provider.allocated, 0U);
}
TEST(NativeQwenMemoryBinding, RejectsForeignAllocationAndReturnsItToProvider) {
  Provider provider;
  provider.foreign = true;
  auto api = Api(provider);
  CapabilityDeviceAllocator allocator(api, 0);
  EXPECT_FALSE(allocator.allocate(32, 128).ok());
  EXPECT_EQ(provider.released, 1U);
}
TEST(NativeQwenMemoryBinding, CopiesOnlyCpuToSelectedDevice) {
  Provider provider;
  auto api = Api(provider);
  CapabilityMemoryCopier copier(api, 0);
  int source{};
  auto destination = reinterpret_cast<void*>(uintptr_t(0x10000));
  EXPECT_TRUE(copier.copy(destination, Device::Create(DeviceType::kCuda, 0).value(),
                         &source, Device::Cpu(), 32).ok());
  EXPECT_FALSE(copier.copy(destination, Device::Cpu(), &source, Device::Cpu(), 32).ok());
  EXPECT_EQ(provider.copies, 1U);
}
TEST(NativeQwenMemoryBinding, RetirementFailureCannotUnloadLiveOwner) {
  Provider provider;
  provider.fail_release = true;
  auto api = Api(provider);
  CapabilityDeviceAllocator allocator(api, 0);
  auto value = allocator.allocate(32, 128);
  ASSERT_TRUE(value.ok());
  EXPECT_DEATH(allocator.deallocate(*value), "");
}
}  // namespace
