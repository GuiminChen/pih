#include "pih/model/deepseek_bound_mhc_stage_work_provider.h"

#include <gtest/gtest.h>

#include <array>

namespace pih {
namespace {

DeepSeekMhcStageSequenceWork item(std::uintptr_t identity,
                                  DeepSeekMhcBranchKind kind,
                                  std::uint32_t layer,
                                  std::uint32_t tokens) {
  DeepSeekMhcStageSequenceWork result;
  result.executor = reinterpret_cast<DeepSeekMhcSequenceExecutor*>(
      identity * 2 + 1);
  result.transaction = reinterpret_cast<DeepSeekAttentionSequenceTransaction*>(
      identity * 2 + 2);
  result.submission.kind = kind;
  result.submission.layer_id = layer;
  result.submission.token_count = tokens;
  return result;
}

TEST(DeepSeekBoundMhcStageWorkProviderTest,
     BindsSeparateAttentionAndFeedForwardBranches) {
  auto provider = DeepSeekBoundMhcStageWorkProvider::Create({5, 5}, 4);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 2, 2};
  const std::array<DeepSeekMhcStageSequenceWork, 2> attention_items{{
      item(1, DeepSeekMhcBranchKind::kAttention, 5, 2),
      item(2, DeepSeekMhcBranchKind::kAttention, 5, 2)}};
  const std::array<DeepSeekMhcStageSequenceWork, 2> feed_items{{
      item(3, DeepSeekMhcBranchKind::kFeedForward, 5, 2),
      item(4, DeepSeekMhcBranchKind::kFeedForward, 5, 2)}};
  const std::array<DeepSeekBoundMhcLayerWork, 1> attention{{
      {5, attention_items}}};
  const std::array<DeepSeekBoundMhcLayerWork, 1> feed{{
      {5, feed_items}}};
  ASSERT_TRUE(provider->bind(descriptor, attention, feed).ok());

  auto attention_result = provider->resolve(
      {DeepSeekStageOperatorKind::kAttention, 5}, descriptor);
  auto feed_result = provider->resolve(
      {DeepSeekStageOperatorKind::kMoe, 5}, descriptor);
  ASSERT_TRUE(attention_result.ok()); ASSERT_TRUE(feed_result.ok());
  EXPECT_EQ(attention_result->front().submission.kind,
            DeepSeekMhcBranchKind::kAttention);
  EXPECT_EQ(feed_result->front().submission.kind,
            DeepSeekMhcBranchKind::kFeedForward);
  EXPECT_FALSE(provider->resolve(
      {DeepSeekStageOperatorKind::kHead, 5}, descriptor).ok());
}

TEST(DeepSeekBoundMhcStageWorkProviderTest,
     FailedBranchRebindPreservesPriorDescriptor) {
  auto provider = DeepSeekBoundMhcStageWorkProvider::Create({5, 5}, 2);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor first{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  const std::array<DeepSeekMhcStageSequenceWork, 1> attention_items{{
      item(1, DeepSeekMhcBranchKind::kAttention, 5, 1)}};
  const std::array<DeepSeekMhcStageSequenceWork, 1> feed_items{{
      item(2, DeepSeekMhcBranchKind::kFeedForward, 5, 1)}};
  const std::array<DeepSeekBoundMhcLayerWork, 1> attention{{
      {5, attention_items}}};
  const std::array<DeepSeekBoundMhcLayerWork, 1> feed{{{5, feed_items}}};
  ASSERT_TRUE(provider->bind(first, attention, feed).ok());
  auto second = first;
  ++second.plan_sequence;
  const std::array<DeepSeekMhcStageSequenceWork, 1> wrong{{
      item(3, DeepSeekMhcBranchKind::kAttention, 5, 1)}};
  const std::array<DeepSeekBoundMhcLayerWork, 1> wrong_feed{{{5, wrong}}};
  EXPECT_FALSE(provider->bind(second, attention, wrong_feed).ok());
  EXPECT_TRUE(provider->resolve(
      {DeepSeekStageOperatorKind::kMoe, 5}, first).ok());
  EXPECT_FALSE(provider->resolve(
      {DeepSeekStageOperatorKind::kMoe, 5}, second).ok());
}

TEST(DeepSeekBoundMhcStageWorkProviderTest,
     RejectsMissingLayerCoverageAndAliasedPackedState) {
  auto provider = DeepSeekBoundMhcStageWorkProvider::Create({5, 6}, 2);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  const std::array<DeepSeekMhcStageSequenceWork, 1> items{{
      item(1, DeepSeekMhcBranchKind::kAttention, 5, 1)}};
  const std::array<DeepSeekBoundMhcLayerWork, 1> incomplete{{{5, items}}};
  EXPECT_FALSE(provider->bind(descriptor, incomplete, incomplete).ok());
}

TEST(DeepSeekBoundMhcStageWorkProviderTest,
     RejectsSyntheticDsparkLayerIdentity) {
  EXPECT_FALSE(DeepSeekBoundMhcStageWorkProvider::Create({43, 43}, 1).ok());
}

}  // namespace
}  // namespace pih
