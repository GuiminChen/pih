#include "pih/model/deepseek_attention_state.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(DeepSeekAttentionStateTest, ReproducesCanonicalLogicalPayloads) {
  EXPECT_EQ(DeepSeekAttentionStateGeometry::MainLogicalBytes(4096).value(),
            46022656U);
  EXPECT_EQ(DeepSeekAttentionStateGeometry::MainLogicalBytes(32768).value(),
            243286016U);
  EXPECT_EQ(DeepSeekAttentionStateGeometry::MainLogicalBytes(131072).value(),
            919617536U);
  EXPECT_EQ(DeepSeekAttentionStateGeometry::MainLogicalBytes(1048576).value(),
            7232045056ULL);
}

TEST(DeepSeekAttentionStateTest, StageOwnershipSumsToWholeModel) {
  for (std::uint32_t world = 1; world <= 4; ++world) {
    auto plan = DeepSeekPipelinePlan::Create(world, true);
    ASSERT_TRUE(plan.ok());
    std::uint64_t total = 0;
    for (std::uint32_t rank = 0; rank < world; ++rank) {
      auto bytes = DeepSeekAttentionStateGeometry::StageLogicalBytes(
          plan->rank(rank), 4096);
      ASSERT_TRUE(bytes.ok());
      total += *bytes;
    }
    EXPECT_EQ(total, 46022656U + 393216U);
  }
}

TEST(DeepSeekAttentionStateTest, ReservationPublishesAndRollsBackAtomically) {
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  ASSERT_TRUE(plan.ok());
  auto bytes = DeepSeekAttentionStateGeometry::StageLogicalBytes(plan->rank(0), 4096);
  ASSERT_TRUE(bytes.ok());
  DeepSeekAttentionStatePool pool(*bytes * 2U);

  auto first = pool.reserve(7, plan->rank(0), 4096);
  ASSERT_TRUE(first.ok()) << first.status().message();
  EXPECT_EQ(pool.reserved_bytes(), *bytes);
  EXPECT_TRUE(first->publish().ok());
  EXPECT_EQ(pool.owned_bytes(), *bytes);
  EXPECT_FALSE(first->rollback().ok());
  EXPECT_TRUE(first->release().ok());
  EXPECT_EQ(pool.available_bytes(), *bytes * 2U);

  auto second = pool.reserve(7, plan->rank(0), 4096);
  ASSERT_TRUE(second.ok());
  EXPECT_GT(second->generation(), first->generation());
  EXPECT_TRUE(second->rollback().ok());
  EXPECT_TRUE(second->rollback().ok());
  EXPECT_EQ(pool.available_bytes(), *bytes * 2U);
}

TEST(DeepSeekAttentionStateTest, RejectsCapacityWithoutPartialReservation) {
  auto plan = DeepSeekPipelinePlan::Create(1, true);
  ASSERT_TRUE(plan.ok());
  DeepSeekAttentionStatePool pool(46022656U);
  auto admission = pool.reserve(1, plan->rank(0), 4096);
  EXPECT_FALSE(admission.ok());
  EXPECT_EQ(admission.status().code(), StatusCode::kResourceExhausted);
  EXPECT_EQ(pool.available_bytes(), 46022656U);
}

TEST(DeepSeekAttentionStateTest, RejectsDuplicateLiveSequenceIdentity) {
  auto plan = DeepSeekPipelinePlan::Create(4, false);
  ASSERT_TRUE(plan.ok());
  auto bytes = DeepSeekAttentionStateGeometry::StageLogicalBytes(plan->rank(0), 4096);
  ASSERT_TRUE(bytes.ok());
  DeepSeekAttentionStatePool pool(*bytes * 2U);
  auto first = pool.reserve(9, plan->rank(0), 4096);
  ASSERT_TRUE(first.ok());
  auto duplicate = pool.reserve(9, plan->rank(0), 4096);
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.status().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(pool.reserved_bytes(), *bytes);
  ASSERT_TRUE(first->rollback().ok());
  EXPECT_TRUE(pool.reserve(9, plan->rank(0), 4096).ok());
}

}  // namespace
}  // namespace pih
