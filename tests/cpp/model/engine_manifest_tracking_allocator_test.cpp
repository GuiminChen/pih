#include "pih/model/engine_manifest_tracking_allocator.h"

#include <array>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest tracking_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

class Clock final : public EngineCudaOwnerLedgerClock {
 public:
  Result<std::uint64_t> monotonic_ns() override { return ++now; }
  std::uint64_t now = 0;
};

class HeapAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++allocate_calls;
    if (!allocate_status.ok()) return allocate_status;
    void* data = std::malloc(static_cast<std::size_t>(bytes));
    return Allocation{data, bytes, alignment, next_generation++, device};
  }
  void deallocate(Allocation allocation) noexcept override {
    ++deallocate_calls;
    released_generation = allocation.generation;
    if (before_free) before_free();
    std::free(allocation.data);
  }
  Status allocate_status = Status::Ok();
  Device device = Device::Create(DeviceType::kCuda, 0).value();
  std::uint64_t next_generation = 10;
  int allocate_calls = 0;
  int deallocate_calls = 0;
  std::uint64_t released_generation = 0;
  std::function<void()> before_free;
};

class PinnedHeapAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++allocate_calls;
    void* data = std::malloc(static_cast<std::size_t>(bytes));
    return Allocation{data, bytes, alignment, next_generation++, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ++deallocate_calls;
    if (before_free) before_free();
    std::free(allocation.data);
  }
  std::uint64_t next_generation = 20;
  int allocate_calls = 0;
  int deallocate_calls = 0;
  std::function<void()> before_free;
};

TEST(EngineManifestTrackingAllocatorTest,
     DeviceAllocationRegistersAndReleaseLeavesTombstone) {
  Clock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  HeapAllocator heap;
  const std::array plans{EngineTrackedGpuAllocationPlan{
      70, tracking_digest(9), 4096, 256}};
  auto allocator =
      EngineTrackedGpuAllocator::Create(plans, heap, registry).value();

  auto allocation = allocator->allocate(4096, 256);
  ASSERT_TRUE(allocation.ok());
  allocator->deallocate(*allocation);
  const std::array<std::uint64_t, 1> pinned{60};
  const std::array<std::uint64_t, 1> allocations{70};
  ASSERT_TRUE(registry.register_pinned(60, tracking_digest(9), 0, 1).ok());
  auto snapshot = registry.capture(pinned, allocations);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_FALSE(snapshot->allocations[0].allocated);
  EXPECT_EQ(snapshot->allocations[0].allocated_bytes, 0U);
  EXPECT_EQ(heap.deallocate_calls, 1);
}

TEST(EngineManifestTrackingAllocatorTest,
     RejectsPlanDriftBeforeCallingUnderlyingAllocator) {
  Clock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  HeapAllocator heap;
  const std::array plans{EngineTrackedGpuAllocationPlan{
      70, tracking_digest(9), 4096, 256}};
  auto allocator =
      EngineTrackedGpuAllocator::Create(plans, heap, registry).value();
  EXPECT_EQ(allocator->allocate(4097, 256).status().code(),
            StatusCode::kFailedPrecondition);
  EXPECT_EQ(heap.allocate_calls, 0);
  EXPECT_EQ(registry.capture(std::span<const std::uint64_t>{},
                             std::span<const std::uint64_t>{})
                .status().code(), StatusCode::kFailedPrecondition);
}

TEST(EngineManifestTrackingAllocatorTest,
     RegistryFailureRollsBackUnderlyingAllocation) {
  Clock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  HeapAllocator heap;
  ASSERT_TRUE(registry.register_allocation(70, tracking_digest(9), 1).ok());
  const std::array plans{EngineTrackedGpuAllocationPlan{
      70, tracking_digest(9), 4096, 256}};
  auto allocator =
      EngineTrackedGpuAllocator::Create(plans, heap, registry).value();
  EXPECT_EQ(allocator->allocate(4096, 256).status().code(),
            StatusCode::kFailedPrecondition);
  EXPECT_EQ(heap.deallocate_calls, 1);
}

TEST(EngineManifestTrackingAllocatorTest,
     WrongReleasePoisonsLedgerWithoutFreeingUnknownAllocation) {
  Clock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  HeapAllocator heap;
  const std::array plans{EngineTrackedGpuAllocationPlan{
      70, tracking_digest(9), 4096, 256}};
  auto allocator =
      EngineTrackedGpuAllocator::Create(plans, heap, registry).value();
  auto allocation = allocator->allocate(4096, 256).value();
  ++allocation.generation;
  allocator->deallocate(allocation);
  EXPECT_EQ(heap.deallocate_calls, 0);
  const std::array<std::uint64_t, 1> allocations{70};
  EXPECT_EQ(registry.capture(std::span<const std::uint64_t>{}, allocations)
                .status().code(), StatusCode::kFailedPrecondition);
}

TEST(EngineManifestTrackingAllocatorTest, RejectsInvalidPlan) {
  Clock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  HeapAllocator heap;
  const std::array plans{EngineTrackedGpuAllocationPlan{
      0, tracking_digest(9), 4096, 256}};
  EXPECT_FALSE(EngineTrackedGpuAllocator::Create(plans, heap, registry).ok());
}

TEST(EngineManifestTrackingAllocatorTest,
     PinnedAllocationRegistersNumaAndReleaseLeavesTombstone) {
  Clock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  PinnedHeapAllocator heap;
  const std::array plans{EngineTrackedPinnedAllocationPlan{
      60, tracking_digest(9), 2, 4096, 256}};
  auto allocator =
      EngineTrackedPinnedAllocator::Create(plans, heap, registry).value();

  auto allocation = allocator->allocate(4096, 256);
  ASSERT_TRUE(allocation.ok());
  const std::array<std::uint64_t, 1> pinned{60};
  const std::array<std::uint64_t, 1> allocations{70};
  ASSERT_TRUE(registry.register_allocation(70, tracking_digest(9), 1).ok());
  auto live = registry.capture(pinned, allocations);
  ASSERT_TRUE(live.ok());
  EXPECT_TRUE(live->pinned[0].registered);
  EXPECT_EQ(live->pinned[0].numa_node, 2);
  EXPECT_EQ(live->pinned[0].registered_bytes, 4096U);

  allocator->deallocate(*allocation);
  auto released = registry.capture(pinned, allocations);
  ASSERT_TRUE(released.ok());
  EXPECT_FALSE(released->pinned[0].registered);
  EXPECT_EQ(released->pinned[0].registered_bytes, 0U);
  EXPECT_EQ(heap.deallocate_calls, 1);
}

TEST(EngineManifestTrackingAllocatorTest,
     PinnedPlanDriftPoisonsBeforeUnderlyingAllocation) {
  Clock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  PinnedHeapAllocator heap;
  const std::array plans{EngineTrackedPinnedAllocationPlan{
      60, tracking_digest(9), 0, 4096, 256}};
  auto allocator =
      EngineTrackedPinnedAllocator::Create(plans, heap, registry).value();
  EXPECT_EQ(allocator->allocate(4096, 128).status().code(),
            StatusCode::kFailedPrecondition);
  EXPECT_EQ(heap.allocate_calls, 0);
  EXPECT_EQ(registry.capture(std::span<const std::uint64_t>{},
                             std::span<const std::uint64_t>{})
                .status().code(), StatusCode::kFailedPrecondition);
}

TEST(EngineManifestTrackingAllocatorTest,
     KeepsOwnersVisibleUntilUnderlyingReleaseReturns) {
  for (int kind = 0; kind < 2; ++kind) {
    Clock clock;
    EngineCudaOwnerLedgerRegistry registry(clock);
    HeapAllocator device_heap;
    PinnedHeapAllocator pinned_heap;
    const std::array device_plans{EngineTrackedGpuAllocationPlan{
        70, tracking_digest(9), 4096, 256}};
    const std::array pinned_plans{EngineTrackedPinnedAllocationPlan{
        60, tracking_digest(9), 2, 4096, 256}};
    auto device = EngineTrackedGpuAllocator::Create(
        device_plans, device_heap, registry).value();
    auto pinned = EngineTrackedPinnedAllocator::Create(
        pinned_plans, pinned_heap, registry).value();
    auto device_allocation = device->allocate(4096, 256).value();
    auto pinned_allocation = pinned->allocate(4096, 256).value();
    const std::array<std::uint64_t, 1> pinned_ids{60};
    const std::array<std::uint64_t, 1> device_ids{70};
    bool visible_during_release = false;
    const auto observe = [&] {
      auto snapshot = registry.capture(pinned_ids, device_ids);
      ASSERT_TRUE(snapshot.ok());
      visible_during_release = kind == 0
                                   ? snapshot->allocations[0].allocated
                                   : snapshot->pinned[0].registered;
    };
    if (kind == 0) {
      device_heap.before_free = observe;
      device->deallocate(device_allocation);
    } else {
      pinned_heap.before_free = observe;
      pinned->deallocate(pinned_allocation);
    }
    EXPECT_TRUE(visible_during_release) << kind;
    if (kind == 0)
      pinned->deallocate(pinned_allocation);
    else
      device->deallocate(device_allocation);
  }
}

TEST(EngineManifestTrackingAllocatorTest, RejectsInvalidPinnedNumaPlan) {
  Clock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  PinnedHeapAllocator heap;
  const std::array plans{EngineTrackedPinnedAllocationPlan{
      60, tracking_digest(9), -1, 4096, 256}};
  EXPECT_FALSE(EngineTrackedPinnedAllocator::Create(plans, heap, registry).ok());
}

}  // namespace
}  // namespace pih
