#include "pih/model/deepseek_rank_plan_resource_pool.h"

#include <gtest/gtest.h>

#include <array>

namespace pih { namespace {

std::array<DeepSeekRankPlanResourceCapacity,
           DeepSeekRankPlanResourcePool::kKindCount>
capacities(std::uint64_t units = 8) {
  std::array<DeepSeekRankPlanResourceCapacity,
             DeepSeekRankPlanResourcePool::kKindCount> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = {
        static_cast<DeepSeekRankPlanResourceKind>(index), units};
  }
  return result;
}

TEST(DeepSeekRankPlanResourcePoolTest, FreezesCompleteTypedCapacitySet) {
  auto values = capacities();
  auto pool = DeepSeekRankPlanResourcePool::Create(7, 1, 2, values);
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  EXPECT_TRUE((*pool)->fully_released());
  EXPECT_EQ((*pool)->capacity(
                DeepSeekRankPlanResourceKind::kAttentionStateTransaction),
            8U);
  auto duplicate = values;
  duplicate.back().kind = duplicate.front().kind;
  EXPECT_FALSE(DeepSeekRankPlanResourcePool::Create(7, 1, 2, duplicate).ok());
}

TEST(DeepSeekRankPlanResourcePoolTest,
     EnforcesPlanIdentityCapacityAndExactReturn) {
  auto values = capacities(2);
  auto pool = DeepSeekRankPlanResourcePool::Create(7, 0, 1, values).value();
  auto stage = DeepSeekPipelinePlan::Create(1, false).value().rank(0);
  DeepSeekPipelinePlanDescriptor plan{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  auto first = pool->reserve(
      plan, stage,
      {DeepSeekRankPlanResourceKind::kOperatorWorkspaceBytes, 2});
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(pool->available(
                DeepSeekRankPlanResourceKind::kOperatorWorkspaceBytes),
            0U);
  EXPECT_FALSE(pool->reserve(
      plan, stage,
      {DeepSeekRankPlanResourceKind::kOperatorWorkspaceBytes, 1}).ok());
  auto wrong_epoch = plan;
  wrong_epoch.engine_epoch = 8;
  EXPECT_FALSE(pool->reserve(
      wrong_epoch, stage,
      {DeepSeekRankPlanResourceKind::kControlSlot, 1}).ok());
  first->reset();
  EXPECT_TRUE(pool->fully_released());
}

TEST(DeepSeekRankPlanResourcePoolTest,
     ReservationFailureReturnsAllPoolCredits) {
  auto values = capacities(8);
  values[static_cast<std::size_t>(
      DeepSeekRankPlanResourceKind::kExpertWorkspaceBytes)].units = 1;
  auto pool = DeepSeekRankPlanResourcePool::Create(7, 0, 1, values).value();
  auto stage = DeepSeekPipelinePlan::Create(1, false).value().rank(0);
  auto reservation = DeepSeekRankPlanReservation::Prepare(
      {7, 9, DeepSeekPlanPhase::kDecode, 1, 1}, stage, 1,
      DeepSeekRoutedExpertResidency::kFullResident, 0, 2, 2, *pool);
  EXPECT_FALSE(reservation.ok());
  EXPECT_TRUE(pool->fully_released());
}

} }  // namespace pih
