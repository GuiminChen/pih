#include "pih/model/deepseek_bound_endpoint_stage_work_provider.h"

#include <gtest/gtest.h>

#include <array>

namespace pih {
namespace {

DeepSeekEndpointStageSequenceWork work(std::uintptr_t identity) {
  DeepSeekEndpointStageSequenceWork value;
  value.executor = reinterpret_cast<DeepSeekEndpointSequenceExecutor*>(
      identity * 2 + 1);
  value.transaction = reinterpret_cast<DeepSeekAttentionSequenceTransaction*>(
      identity * 2 + 2);
  return value;
}

TEST(DeepSeekBoundEndpointStageWorkProviderTest,
     BindsExactPlanAndHonorsStageEndpointOwnership) {
  auto provider = DeepSeekBoundEndpointStageWorkProvider::Create(4, true, false);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 2, 2};
  const std::array<DeepSeekEndpointStageSequenceWork, 2> packed{{
      work(1), work(2)}};
  ASSERT_TRUE(provider->bind(descriptor, packed).ok());
  auto embedding = provider->resolve(
      {DeepSeekStageOperatorKind::kEmbedding, 0}, descriptor);
  ASSERT_TRUE(embedding.ok());
  EXPECT_EQ(embedding->size(), 2U);
  EXPECT_FALSE(provider->resolve(
      {DeepSeekStageOperatorKind::kHead, 0}, descriptor).ok());
  auto stale = descriptor;
  ++stale.plan_sequence;
  EXPECT_FALSE(provider->resolve(
      {DeepSeekStageOperatorKind::kEmbedding, 0}, stale).ok());
  EXPECT_FALSE(provider->resolve(
      {DeepSeekStageOperatorKind::kEmbedding, 1}, descriptor).ok());
}

TEST(DeepSeekBoundEndpointStageWorkProviderTest,
     FailedAliasingRebindPreservesPreviouslyPublishedPlan) {
  auto provider = DeepSeekBoundEndpointStageWorkProvider::Create(2, true, true);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor first{
      7, 9, DeepSeekPlanPhase::kDecode, 2, 2};
  const std::array<DeepSeekEndpointStageSequenceWork, 2> valid{{
      work(1), work(2)}};
  ASSERT_TRUE(provider->bind(first, valid).ok());
  auto second = first;
  ++second.plan_sequence;
  const std::array<DeepSeekEndpointStageSequenceWork, 2> aliased{{
      work(3), work(3)}};
  EXPECT_FALSE(provider->bind(second, aliased).ok());
  EXPECT_TRUE(provider->resolve(
      {DeepSeekStageOperatorKind::kHead, 0}, first).ok());
  EXPECT_FALSE(provider->resolve(
      {DeepSeekStageOperatorKind::kHead, 0}, second).ok());
}

TEST(DeepSeekBoundEndpointStageWorkProviderTest,
     RejectsCapacityAndPackedCountMismatch) {
  EXPECT_FALSE(DeepSeekBoundEndpointStageWorkProvider::Create(
      0, true, false).ok());
  EXPECT_FALSE(DeepSeekBoundEndpointStageWorkProvider::Create(
      1, false, false).ok());
  auto provider = DeepSeekBoundEndpointStageWorkProvider::Create(2, false, true);
  ASSERT_TRUE(provider.ok());
  const std::array<DeepSeekEndpointStageSequenceWork, 1> packed{{work(1)}};
  EXPECT_FALSE(provider->bind(
      {7, 9, DeepSeekPlanPhase::kDecode, 1, 2}, packed).ok());
}

}  // namespace
}  // namespace pih
