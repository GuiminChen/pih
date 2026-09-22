#include "pih/model/deepseek_bound_dense_attention_work_provider.h"

#include <gtest/gtest.h>

#include <array>

namespace pih {
namespace {

DeepSeekDenseAttentionStageSequenceWork item(std::uintptr_t identity,
                                             std::uint32_t tokens) {
  DeepSeekDenseAttentionStageSequenceWork value;
  value.input_coordinator =
      reinterpret_cast<DeepSeekAttentionProjectionCoordinator*>(
          identity * 3 + 1);
  value.output_coordinator =
      reinterpret_cast<DeepSeekAttentionOutputProjectionCoordinator*>(
          identity * 3 + 2);
  value.transaction = reinterpret_cast<DeepSeekAttentionSequenceTransaction*>(
      identity * 3 + 3);
  value.input.input_quant.token_count = tokens;
  value.sparse_query_bf16 = identity * 16 + 1;
  value.sparse_kv_bf16 = identity * 16 + 2;
  value.sparse_output_bf16 = identity * 16 + 3;
  return value;
}

TEST(DeepSeekBoundDenseAttentionWorkProviderTest,
     BindsEveryLayerAndExactPackedTokenTotal) {
  auto provider = DeepSeekBoundDenseAttentionWorkProvider::Create({5, 6}, 4);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kPrefill, 5, 2};
  const std::array<DeepSeekDenseAttentionStageSequenceWork, 2> layer5{{
      item(1, 2), item(2, 3)}};
  const std::array<DeepSeekDenseAttentionStageSequenceWork, 2> layer6{{
      item(3, 1), item(4, 4)}};
  const std::array<DeepSeekBoundDenseAttentionLayerWork, 2> layers{{
      {6, layer6}, {5, layer5}}};
  ASSERT_TRUE(provider->bind(descriptor, layers).ok());
  auto resolved = provider->resolve(
      {DeepSeekStageOperatorKind::kAttention, 5}, descriptor);
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(resolved->size(), 2U);
  auto stale = descriptor;
  ++stale.plan_sequence;
  EXPECT_FALSE(provider->resolve(
      {DeepSeekStageOperatorKind::kAttention, 5}, stale).ok());
  EXPECT_FALSE(provider->resolve(
      {DeepSeekStageOperatorKind::kMoe, 5}, descriptor).ok());
}

TEST(DeepSeekBoundDenseAttentionWorkProviderTest,
     FailedTokenTotalRebindPreservesPriorPlan) {
  auto provider = DeepSeekBoundDenseAttentionWorkProvider::Create({5, 5}, 2);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor first{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  const std::array<DeepSeekDenseAttentionStageSequenceWork, 1> valid{{
      item(1, 1)}};
  const std::array<DeepSeekBoundDenseAttentionLayerWork, 1> first_layers{{
      {5, valid}}};
  ASSERT_TRUE(provider->bind(first, first_layers).ok());
  auto second = first;
  ++second.plan_sequence;
  const std::array<DeepSeekDenseAttentionStageSequenceWork, 1> wrong{{
      item(2, 2)}};
  const std::array<DeepSeekBoundDenseAttentionLayerWork, 1> wrong_layers{{
      {5, wrong}}};
  EXPECT_FALSE(provider->bind(second, wrong_layers).ok());
  EXPECT_TRUE(provider->resolve(
      {DeepSeekStageOperatorKind::kAttention, 5}, first).ok());
  EXPECT_FALSE(provider->resolve(
      {DeepSeekStageOperatorKind::kAttention, 5}, second).ok());
}

TEST(DeepSeekBoundDenseAttentionWorkProviderTest,
     RejectsAliasedSequenceStateAndMissingLayerCoverage) {
  auto provider = DeepSeekBoundDenseAttentionWorkProvider::Create({5, 6}, 2);
  ASSERT_TRUE(provider.ok());
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 2, 2};
  const auto shared = item(1, 1);
  const std::array<DeepSeekDenseAttentionStageSequenceWork, 2> aliased{{
      shared, shared}};
  const std::array<DeepSeekBoundDenseAttentionLayerWork, 1> missing{{
      {5, aliased}}};
  EXPECT_FALSE(provider->bind(descriptor, missing).ok());
}

TEST(DeepSeekBoundDenseAttentionWorkProviderTest,
     RejectsSyntheticDsparkLayerIdentity) {
  EXPECT_FALSE(DeepSeekBoundDenseAttentionWorkProvider::Create(
      {43, 43}, 1).ok());
}

}  // namespace
}  // namespace pih
