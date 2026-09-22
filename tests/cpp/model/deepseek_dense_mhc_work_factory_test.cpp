#include "pih/model/deepseek_dense_mhc_work_factory.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

template <typename T>
T* fake(std::uintptr_t value) {
  return reinterpret_cast<T*>(value);
}

DeepSeekDenseAttentionLayerPlanInput dense_layer(std::uint32_t layer) {
  DeepSeekDenseAttentionLayerPlanInput result;
  result.layer = layer;
  for (std::uint32_t sequence = 0; sequence < 2; ++sequence) {
    DeepSeekDenseAttentionStageSequenceWork work;
    work.input_coordinator = fake<DeepSeekAttentionProjectionCoordinator>(
        0x10000 + layer * 0x100 + sequence * 0x10);
    work.output_coordinator =
        fake<DeepSeekAttentionOutputProjectionCoordinator>(
            0x20000 + layer * 0x100 + sequence * 0x10);
    work.transaction = fake<DeepSeekAttentionSequenceTransaction>(
        0x30000 + layer * 0x100 + sequence * 0x10);
    work.input.input_quant.token_count = sequence + 1;
    work.sparse_query_bf16 = 1;
    work.sparse_kv_bf16 = 2;
    work.sparse_output_bf16 = 3;
    result.sequences.push_back(work);
  }
  return result;
}

DeepSeekMhcLayerPlanInput mhc_layer(
    std::uint32_t layer, DeepSeekMhcBranchKind kind) {
  DeepSeekMhcLayerPlanInput result;
  result.layer = layer;
  for (std::uint32_t sequence = 0; sequence < 2; ++sequence) {
    DeepSeekMhcStageSequenceWork work;
    work.executor = fake<DeepSeekMhcSequenceExecutor>(
        0x40000 + layer * 0x100 + sequence * 0x10 +
        (kind == DeepSeekMhcBranchKind::kFeedForward ? 0x10000 : 0));
    work.transaction = fake<DeepSeekAttentionSequenceTransaction>(
        0x60000 + layer * 0x100 + sequence * 0x10 +
        (kind == DeepSeekMhcBranchKind::kFeedForward ? 0x10000 : 0));
    work.submission.kind = kind;
    work.submission.layer_id = layer;
    work.submission.token_count = 3;
    result.sequences.push_back(work);
  }
  return result;
}

TEST(DeepSeekDenseMhcWorkFactoryTest,
     AtomicallyPublishesDenseAndBothMhcBranches) {
  auto factory = DeepSeekDenseMhcWorkFactory::Create({5, 6}, 2, 8).value();
  DeepSeekRankComputeWorkBuilder builder;
  ASSERT_TRUE(factory.append_plan_work(
      3, 2, {dense_layer(5), dense_layer(6)},
      {mhc_layer(5, DeepSeekMhcBranchKind::kAttention),
       mhc_layer(6, DeepSeekMhcBranchKind::kAttention)},
      {mhc_layer(5, DeepSeekMhcBranchKind::kFeedForward),
       mhc_layer(6, DeepSeekMhcBranchKind::kFeedForward)}, builder).ok());
  auto work = std::move(builder).finish().value();
  EXPECT_EQ(work.dense_attention.size(), 2U);
  EXPECT_EQ(work.mhc_attention.size(), 2U);
  EXPECT_EQ(work.mhc_feed_forward.size(), 2U);
  EXPECT_EQ(work.mhc_feed_forward[1].sequences[0].submission.kind,
            DeepSeekMhcBranchKind::kFeedForward);
}

TEST(DeepSeekDenseMhcWorkFactoryTest,
     RejectsTokenDriftCoverageGapsAndAliasedSequenceState) {
  auto factory = DeepSeekDenseMhcWorkFactory::Create({5, 6}, 2, 8).value();
  DeepSeekRankComputeWorkBuilder missing;
  EXPECT_FALSE(factory.append_plan_work(
      3, 2, {dense_layer(5)}, {}, {}, missing).ok());

  auto dense5 = dense_layer(5);
  auto dense6 = dense_layer(6);
  dense6.sequences[1].input.input_quant.token_count = 1;
  DeepSeekRankComputeWorkBuilder drift;
  EXPECT_FALSE(factory.append_plan_work(
      3, 2, {std::move(dense5), std::move(dense6)},
      {mhc_layer(5, DeepSeekMhcBranchKind::kAttention),
       mhc_layer(6, DeepSeekMhcBranchKind::kAttention)},
      {mhc_layer(5, DeepSeekMhcBranchKind::kFeedForward),
       mhc_layer(6, DeepSeekMhcBranchKind::kFeedForward)}, drift).ok());

  auto aliased = mhc_layer(5, DeepSeekMhcBranchKind::kAttention);
  aliased.sequences[1].executor = aliased.sequences[0].executor;
  DeepSeekRankComputeWorkBuilder alias;
  EXPECT_FALSE(factory.append_plan_work(
      3, 2, {dense_layer(5), dense_layer(6)},
      {std::move(aliased),
       mhc_layer(6, DeepSeekMhcBranchKind::kAttention)},
      {mhc_layer(5, DeepSeekMhcBranchKind::kFeedForward),
       mhc_layer(6, DeepSeekMhcBranchKind::kFeedForward)}, alias).ok());
}

}  // namespace
}  // namespace pih
