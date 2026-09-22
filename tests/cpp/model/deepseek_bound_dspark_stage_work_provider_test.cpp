#include "pih/model/deepseek_bound_dspark_stage_work_provider.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekDsparkStageWork work(std::uintptr_t identity) {
  DeepSeekDsparkStageWork value;
  value.embed_coordinator = reinterpret_cast<DeepSeekDsparkEmbedCoordinator*>(
      identity * 4 + 1);
  value.head_executor = reinterpret_cast<DeepSeekDsparkHeadExecutor*>(
      identity * 4 + 2);
  value.transaction = reinterpret_cast<DeepSeekAttentionSequenceTransaction*>(
      identity * 4 + 3);
  return value;
}

TEST(DeepSeekBoundDsparkStageWorkProviderTest,
     BindsCompleteWorkForExactDecodePlan) {
  auto provider = DeepSeekBoundDsparkStageWorkProvider::Create(true, 8);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 2, 2};
  ASSERT_TRUE(provider->bind(descriptor, work(1)).ok());
  auto resolved = provider->resolve(descriptor);
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ((*resolved)->head_executor, work(1).head_executor);
  auto stale = descriptor;
  ++stale.plan_sequence;
  EXPECT_FALSE(provider->resolve(stale).ok());

  const DeepSeekPipelinePlanDescriptor prefill{
      7, 10, DeepSeekPlanPhase::kPrefill, 4, 1};
  auto prefill_work = work(2);
  prefill_work.kind =
      DeepSeekDsparkStageWorkKind::kPrefillStateInitialization;
  prefill_work.head_executor = nullptr;
  ASSERT_TRUE(provider->bind(prefill, prefill_work).ok());
  EXPECT_FALSE(provider->resolve(descriptor).ok());
  auto resolved_prefill = provider->resolve(prefill);
  ASSERT_TRUE(resolved_prefill.ok());
  EXPECT_EQ((*resolved_prefill)->head_executor, nullptr);
}

TEST(DeepSeekBoundDsparkStageWorkProviderTest,
     FailedRebindPreservesPriorPublishedPlan) {
  auto provider = DeepSeekBoundDsparkStageWorkProvider::Create(true, 2);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor first{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  ASSERT_TRUE(provider->bind(first, work(1)).ok());
  auto second = first;
  ++second.plan_sequence;
  auto incomplete = work(2);
  incomplete.head_executor = nullptr;
  EXPECT_FALSE(provider->bind(second, incomplete).ok());
  EXPECT_TRUE(provider->resolve(first).ok());
  EXPECT_FALSE(provider->resolve(second).ok());
}

TEST(DeepSeekBoundDsparkStageWorkProviderTest,
     RejectsNonOwnerDrainVerifyAndCapacityOverflow) {
  EXPECT_FALSE(DeepSeekBoundDsparkStageWorkProvider::Create(false, 1).ok());
  EXPECT_FALSE(DeepSeekBoundDsparkStageWorkProvider::Create(true, 0).ok());
  auto provider = DeepSeekBoundDsparkStageWorkProvider::Create(true, 1);
  ASSERT_TRUE(provider.ok());
  EXPECT_FALSE(provider->bind(
      {7, 9, DeepSeekPlanPhase::kVerify, 1, 1}, work(1)).ok());
  EXPECT_FALSE(provider->bind(
      {7, 9, DeepSeekPlanPhase::kDrain, 0, 1}, work(1)).ok());
  EXPECT_FALSE(provider->bind(
      {7, 9, DeepSeekPlanPhase::kDecode, 2, 2}, work(1)).ok());
}

}  // namespace
}  // namespace pih
