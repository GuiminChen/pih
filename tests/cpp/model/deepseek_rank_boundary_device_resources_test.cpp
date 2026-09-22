#include "pih/model/deepseek_rank_boundary_device_resources.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>

namespace pih {
namespace {

class BoundaryResourceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("allocation failed");
    ++active;
    return Allocation{data, bytes, alignment, ++generation,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
    --active;
  }
  std::uint64_t generation = 0;
  int active = 0;
};

class BoundaryResourceDriver final : public CudaRuntimeResourceDriver {
 public:
  Result<std::uintptr_t> retain_primary_context(std::int32_t,
                                                std::uint32_t) override {
    return 77;
  }
  Status bind_runtime(std::int32_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t) override { return 1; }
  Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t context) override {
    if (context != 77 || (fail_after >= 0 && created >= fail_after)) {
      return Status::Internal("event creation failed");
    }
    ++created;
    return static_cast<DriverEventHandle>(100 + created);
  }
  void destroy_event(DriverEventHandle) noexcept override { ++destroyed; }
  void destroy_stream(DriverStreamHandle) noexcept override {}
  void release_primary_context(std::int32_t,
                               std::uintptr_t) noexcept override {}
  int fail_after = -1;
  int created = 0;
  int destroyed = 0;
};

TEST(DeepSeekRankBoundaryDeviceResourcesTest, Pp1OwnsZeroBoundaryResources) {
  BoundaryResourceAllocator allocator;
  BoundaryResourceDriver driver;
  auto resources = DeepSeekRankBoundaryDeviceResources::Allocate(
      0, 1, 8, 77, allocator, driver);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ((*resources)->incoming_slot_count(), 0U);
  EXPECT_EQ((*resources)->incoming_event_count(), 0U);
  EXPECT_EQ((*resources)->outgoing_event_count(), 0U);
  EXPECT_EQ((*resources)->outgoing_warmup_buffer(), nullptr);
  EXPECT_EQ(allocator.active, 0);
  EXPECT_EQ(driver.created, 0);
}

TEST(DeepSeekRankBoundaryDeviceResourcesTest,
     MiddleRankOwnsTwoReceiveSlotsOutgoingWarmupAndFourEvents) {
  BoundaryResourceAllocator allocator;
  BoundaryResourceDriver driver;
  {
    auto resources = DeepSeekRankBoundaryDeviceResources::Allocate(
        1, 3, 1, 77, allocator, driver);
    ASSERT_TRUE(resources.ok()) << resources.status().message();
    EXPECT_EQ((*resources)->slot_bytes(), 32768U);
    EXPECT_EQ((*resources)->incoming_slot_count(), 2U);
    EXPECT_EQ((*resources)->incoming_event_count(), 2U);
    EXPECT_EQ((*resources)->outgoing_event_count(), 2U);
    EXPECT_NE((*resources)->incoming_slot(0), nullptr);
    EXPECT_NE((*resources)->incoming_event(0), 0U);
    EXPECT_NE((*resources)->outgoing_event(1), 0U);
    ASSERT_NE((*resources)->outgoing_warmup_buffer(), nullptr);
    EXPECT_EQ((*resources)->outgoing_warmup_buffer()->size_bytes(), 32768U);
    EXPECT_EQ(allocator.active, 3);
    EXPECT_EQ(driver.created, 4);
  }
  EXPECT_EQ(allocator.active, 0);
  EXPECT_EQ(driver.destroyed, 4);
}

TEST(DeepSeekRankBoundaryDeviceResourcesTest,
     LastRankHasNoOutgoingWarmupBuffer) {
  BoundaryResourceAllocator allocator;
  BoundaryResourceDriver driver;
  auto resources = DeepSeekRankBoundaryDeviceResources::Allocate(
      2, 3, 1, 77, allocator, driver);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ((*resources)->outgoing_warmup_buffer(), nullptr);
  EXPECT_EQ(allocator.active, 2);
}

TEST(DeepSeekRankBoundaryDeviceResourcesTest,
     RollsBackPartialEventsAndRejectsOverflow) {
  BoundaryResourceAllocator allocator;
  BoundaryResourceDriver driver;
  driver.fail_after = 2;
  auto failed = DeepSeekRankBoundaryDeviceResources::Allocate(
      1, 3, 1, 77, allocator, driver);
  EXPECT_FALSE(failed.ok());
  EXPECT_EQ(allocator.active, 0);
  EXPECT_EQ(driver.destroyed, 2);

  auto overflow = DeepSeekRankBoundaryDeviceResources::Allocate(
      0, 2, std::numeric_limits<std::uint64_t>::max(), 77, allocator,
      driver);
  EXPECT_FALSE(overflow.ok());
}

}  // namespace
}  // namespace pih
