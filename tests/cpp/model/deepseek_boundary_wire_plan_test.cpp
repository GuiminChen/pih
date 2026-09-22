#include "pih/model/deepseek_boundary_wire_plan.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class WirePlanAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    return Allocation{data, bytes, alignment, ++generation,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
  std::uint64_t generation = 0;
};

DeepSeekBoundarySendSource source(Buffer& buffer, std::uint32_t tokens,
                                  std::uint64_t owner) {
  return DeepSeekBoundarySendSource::Create(
      buffer, 0, std::uint64_t{tokens} * 32768U, tokens, owner, 77, 9).value();
}

TEST(DeepSeekBoundaryWirePlanTest, JoinsEveryPp3OutgoingSource) {
  WirePlanAllocator allocator;
  auto first = Buffer::Allocate(allocator, 65536, 256).value();
  auto second = Buffer::Allocate(allocator, 65536, 256).value();
  std::vector<DeepSeekRankComputePlanWork> work(3);
  work[0].outgoing_boundary = source(first, 2, 10);
  work[1].outgoing_boundary = source(second, 2, 11);
  auto plan = DeepSeekBoundaryWirePlan::Create(
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 1}, 3, work);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->wire_token_count(), 2U);
  EXPECT_EQ(plan->boundary_count(), 2U);
}

TEST(DeepSeekBoundaryWirePlanTest, DrainRetainsOriginalPositiveWireCount) {
  WirePlanAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 98304, 256).value();
  std::vector<DeepSeekRankComputePlanWork> work(2);
  work[0].outgoing_boundary = source(buffer, 3, 10);
  auto plan = DeepSeekBoundaryWirePlan::Create(
      {3, 2, DeepSeekPlanPhase::kDrain, 0, 1}, 2, work);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->wire_token_count(), 3U);
}

TEST(DeepSeekBoundaryWirePlanTest, RejectsMissingDriftAndLastRankSource) {
  WirePlanAllocator allocator;
  auto first = Buffer::Allocate(allocator, 65536, 256).value();
  auto second = Buffer::Allocate(allocator, 98304, 256).value();
  std::vector<DeepSeekRankComputePlanWork> work(3);
  work[0].outgoing_boundary = source(first, 2, 10);
  EXPECT_FALSE(DeepSeekBoundaryWirePlan::Create(
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 1}, 3, work).ok());
  work[1].outgoing_boundary = source(second, 3, 11);
  EXPECT_FALSE(DeepSeekBoundaryWirePlan::Create(
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 1}, 3, work).ok());
  work[1].outgoing_boundary = source(first, 2, 12);
  work[2].outgoing_boundary = source(first, 2, 13);
  EXPECT_FALSE(DeepSeekBoundaryWirePlan::Create(
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 1}, 3, work).ok());
}

TEST(DeepSeekBoundaryWirePlanTest, Pp1CompilesToZeroWireOperations) {
  std::vector<DeepSeekRankComputePlanWork> work(1);
  auto plan = DeepSeekBoundaryWirePlan::Create(
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 1}, 1, work);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan->wire_token_count(), 0U);
  EXPECT_EQ(plan->boundary_count(), 0U);
}

}  // namespace
}  // namespace pih
