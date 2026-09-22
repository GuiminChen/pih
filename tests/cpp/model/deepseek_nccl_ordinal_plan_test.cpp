#include "pih/model/deepseek_nccl_ordinal_plan.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>

namespace pih {
namespace {

TEST(DeepSeekNcclOrdinalPlanTest, Pp3SeparatesGlobalAndRankLocalOrdinals) {
  auto edge0_send = DeepSeekNcclOrdinalPlan::Create(
      3, 1, 0, DeepSeekNcclRole::kSend);
  auto edge0_recv = DeepSeekNcclOrdinalPlan::Create(
      3, 1, 0, DeepSeekNcclRole::kRecv);
  auto edge1_send = DeepSeekNcclOrdinalPlan::Create(
      3, 1, 1, DeepSeekNcclRole::kSend);
  auto edge1_recv = DeepSeekNcclOrdinalPlan::Create(
      3, 1, 1, DeepSeekNcclRole::kRecv);
  ASSERT_TRUE(edge0_send.ok()); ASSERT_TRUE(edge0_recv.ok());
  ASSERT_TRUE(edge1_send.ok()); ASSERT_TRUE(edge1_recv.ok());
  EXPECT_EQ(edge0_send->global_issue_ordinal, 1U);
  EXPECT_EQ(edge0_recv->global_issue_ordinal, 1U);
  EXPECT_EQ(edge0_send->rank_local_ordinal, 1U);
  EXPECT_EQ(edge0_recv->rank_local_ordinal, 1U);
  EXPECT_EQ(edge1_send->global_issue_ordinal, 2U);
  EXPECT_EQ(edge1_recv->global_issue_ordinal, 2U);
  EXPECT_EQ(edge1_send->rank_local_ordinal, 2U);
  EXPECT_EQ(edge1_recv->rank_local_ordinal, 1U);

  auto next_edge0_send = DeepSeekNcclOrdinalPlan::Create(
      3, 2, 0, DeepSeekNcclRole::kSend).value();
  auto next_edge0_recv = DeepSeekNcclOrdinalPlan::Create(
      3, 2, 0, DeepSeekNcclRole::kRecv).value();
  EXPECT_EQ(next_edge0_send.global_issue_ordinal, 3U);
  EXPECT_EQ(next_edge0_send.rank_local_ordinal, 2U);
  EXPECT_EQ(next_edge0_recv.rank_local_ordinal, 3U);
}

TEST(DeepSeekNcclOrdinalPlanTest, Pp2AndPp4RankProjectionsAreConsecutive) {
  for (std::uint32_t world_size : {2U, 4U}) {
    std::array<std::uint64_t, 4> next{1, 1, 1, 1};
    for (std::uint64_t plan = 1; plan <= 3; ++plan) {
      for (std::uint32_t edge = 0; edge + 1 < world_size; ++edge) {
        auto send = DeepSeekNcclOrdinalPlan::Create(
            world_size, plan, edge, DeepSeekNcclRole::kSend).value();
        auto recv = DeepSeekNcclOrdinalPlan::Create(
            world_size, plan, edge, DeepSeekNcclRole::kRecv).value();
        EXPECT_EQ(send.rank_local_ordinal, next[edge]++);
        EXPECT_EQ(recv.rank_local_ordinal, next[edge + 1]++);
        EXPECT_EQ(send.global_issue_ordinal, recv.global_issue_ordinal);
      }
    }
  }
}

TEST(DeepSeekNcclOrdinalPlanTest, RejectsInvalidTopologyAndOverflow) {
  EXPECT_FALSE(DeepSeekNcclOrdinalPlan::Create(
      1, 1, 0, DeepSeekNcclRole::kSend).ok());
  EXPECT_FALSE(DeepSeekNcclOrdinalPlan::Create(
      3, 1, 2, DeepSeekNcclRole::kSend).ok());
  EXPECT_FALSE(DeepSeekNcclOrdinalPlan::Create(
      4, std::numeric_limits<std::uint64_t>::max(), 0,
      DeepSeekNcclRole::kSend).ok());
}

}  // namespace
}  // namespace pih
