#include "pih/model/deepseek_pipeline_transaction.h"

#include <gtest/gtest.h>

#include <limits>

namespace pih {
namespace {

TEST(DeepSeekPipelineCapacityTest, DerivesOneSharedWireAndExpertCeiling) {
  auto capacity = DeepSeekPipelineCapacity::Create(4, 512, 32, 32, true);
  ASSERT_TRUE(capacity.ok()) << capacity.status().message();
  EXPECT_EQ(capacity->max_pipeline_tokens, 512U);
  EXPECT_EQ(capacity->expert_tokens, 512U);
  EXPECT_EQ(capacity->boundary_slot_bytes, 16U * 1024U * 1024U);
  EXPECT_EQ(capacity->control_payload_bytes, 4352U);
  EXPECT_EQ(capacity->expert_workspace_bytes, 16480U * 512U + 1280U);
  EXPECT_EQ(capacity->rank(0).recv_bytes, 0U);
  EXPECT_EQ(capacity->rank(1).recv_bytes, 32U * 1024U * 1024U);
  EXPECT_EQ(capacity->rank(3).output_hold_bytes, 0U);
  EXPECT_EQ(capacity->rank(0).output_hold_bytes, 32U * 1024U * 1024U);
}

TEST(DeepSeekPipelineCapacityTest, HandlesPp1AndRejectsInvalidEnvelope) {
  auto pp1 = DeepSeekPipelineCapacity::Create(1, 128, 16, 8, false);
  ASSERT_TRUE(pp1.ok());
  EXPECT_EQ(pp1->max_pipeline_tokens, 128U);
  EXPECT_EQ(pp1->rank(0).recv_bytes, 0U);
  EXPECT_EQ(pp1->rank(0).output_hold_bytes, 0U);
  EXPECT_FALSE(DeepSeekPipelineCapacity::Create(0, 1, 1, 1, false).ok());
  EXPECT_FALSE(DeepSeekPipelineCapacity::Create(2, 0, 1, 1, false).ok());
  EXPECT_FALSE(DeepSeekPipelineCapacity::Create(
      2, 1, 1, std::numeric_limits<std::uint32_t>::max(), true).ok());
}

TEST(DeepSeekPipelineTransactionTest, AtomicallyCommitsAcrossAllRanks) {
  auto capacity = DeepSeekPipelineCapacity::Create(3, 128, 8, 8, false);
  ASSERT_TRUE(capacity.ok());
  auto pools = DeepSeekPipelineResourceSet::Create(*capacity);
  ASSERT_TRUE(pools.ok());
  auto transaction = pools->prepare({7, 1, DeepSeekPlanPhase::kDecode, 8, 8});
  ASSERT_TRUE(transaction.ok()) << transaction.status().message();
  EXPECT_EQ(transaction->state(), DeepSeekPipelineTransactionState::kPrepared);
  EXPECT_EQ(pools->rank(1).incoming_available(), 1U);
  EXPECT_EQ(pools->rank(1).outgoing_available(), 1U);
  ASSERT_TRUE(transaction->commit().ok());
  EXPECT_FALSE(transaction->abort_prepare().ok());
  ASSERT_TRUE(transaction->complete().ok());
  EXPECT_EQ(pools->rank(1).incoming_available(), 2U);
  EXPECT_EQ(pools->rank(1).outgoing_available(), 2U);
}

TEST(DeepSeekPipelineTransactionTest, RollsBackEveryRankOnPrepareFailure) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 16, 4, 4, false);
  ASSERT_TRUE(capacity.ok());
  auto pools = DeepSeekPipelineResourceSet::Create(*capacity);
  ASSERT_TRUE(pools.ok());
  pools->rank(1).set_prepare_failure_for_test(true);
  auto transaction = pools->prepare({3, 1, DeepSeekPlanPhase::kPrefill, 4, 1});
  ASSERT_FALSE(transaction.ok());
  EXPECT_EQ(pools->rank(0).outgoing_available(), 2U);
  EXPECT_EQ(pools->rank(0).control_available(), 2U);
  EXPECT_EQ(pools->rank(1).incoming_available(), 2U);
}

TEST(DeepSeekPipelineTransactionTest, EnforcesStrictPlanOrderAndBounds) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 16, 4, 4, false);
  ASSERT_TRUE(capacity.ok());
  auto pools = DeepSeekPipelineResourceSet::Create(*capacity);
  ASSERT_TRUE(pools.ok());
  EXPECT_FALSE(pools->prepare({2, 1, DeepSeekPlanPhase::kDecode, 5, 5}).ok());
  EXPECT_FALSE(pools->prepare({2, 1, DeepSeekPlanPhase::kDecode, 17, 4}).ok());
  auto first = pools->prepare({2, 1, DeepSeekPlanPhase::kDecode, 4, 4});
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(first->abort_prepare().ok());
  EXPECT_FALSE(pools->prepare({2, 1, DeepSeekPlanPhase::kDecode, 4, 4}).ok());
  EXPECT_TRUE(pools->prepare({2, 2, DeepSeekPlanPhase::kDecode, 4, 4}).ok());
}

TEST(DeepSeekPipelineTransactionTest, EnforcesPhaseSpecificTokenBounds) {
  auto capacity = DeepSeekPipelineCapacity::Create(1, 16, 4, 2, true);
  ASSERT_TRUE(capacity.ok());
  auto pools = DeepSeekPipelineResourceSet::Create(*capacity);
  ASSERT_TRUE(pools.ok());
  EXPECT_FALSE(pools->prepare({1, 1, DeepSeekPlanPhase::kPrefill, 17, 1}).ok());
  EXPECT_FALSE(pools->prepare({1, 1, DeepSeekPlanPhase::kDecode, 4, 3}).ok());
  EXPECT_FALSE(pools->prepare({1, 1, DeepSeekPlanPhase::kVerify, 11, 2}).ok());
  auto verify = pools->prepare({1, 1, DeepSeekPlanPhase::kVerify, 10, 2});
  ASSERT_TRUE(verify.ok());
  ASSERT_TRUE(verify->abort_prepare().ok());
  auto drain = pools->prepare({1, 2, DeepSeekPlanPhase::kDrain, 0, 1});
  ASSERT_TRUE(drain.ok());
}

}  // namespace
}  // namespace pih
