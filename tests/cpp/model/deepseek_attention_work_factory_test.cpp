#include "pih/model/deepseek_attention_work_factory.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

template <typename T>
T* fake(std::uintptr_t address) {
  return reinterpret_cast<T*>(address);
}

DeepSeekAttentionLayerRuntimeResources resources(std::uint32_t layer) {
  return {
      layer,
      fake<DeepSeekRecentStateWriter>(0x1000 + layer * 0x100),
      fake<DeepSeekCompressedLayerUpdateCoordinator>(
          0x2000 + layer * 0x100),
      fake<DeepSeekAttentionLayerCoordinator>(0x3000 + layer * 0x100),
      fake<DeepSeekPrefillLayerCoordinator>(0x4000 + layer * 0x100)};
}

DeepSeekDecodeAttentionLayerPlanInput decode_input(
    std::uint32_t layer, std::vector<std::uint32_t>& positions,
    std::vector<std::uint32_t>& visible) {
  DeepSeekDecodeAttentionLayerPlanInput input;
  input.layer = layer;
  input.attention.query_positions = positions;
  input.attention.selection.visible_slot_counts = visible;
  input.transaction = fake<DeepSeekAttentionSequenceTransaction>(
      0x5000 + layer * 0x100);
  return input;
}

DeepSeekChunkAttentionLayerPlanInput chunk_input(
    std::uint32_t layer,
    std::vector<DeepSeekRecentStateSubmission>& recent,
    std::vector<DeepSeekCompressedLayerUpdateSubmission>& updates,
    std::vector<std::uint32_t>& positions,
    std::vector<std::uint32_t>& visible) {
  DeepSeekChunkAttentionLayerPlanInput input;
  input.layer = layer;
  input.submission.recent = recent;
  input.submission.updates = updates;
  input.submission.attention.query_positions = positions;
  input.submission.attention.selection.visible_slot_counts = visible;
  input.transaction = fake<DeepSeekAttentionSequenceTransaction>(
      0x6000 + layer * 0x100);
  return input;
}

TEST(DeepSeekAttentionWorkFactoryTest,
     PublishesDecodeResourcesForEveryOwnedLayer) {
  auto factory = DeepSeekAttentionWorkFactory::Create(
      {10, 11}, {resources(10), resources(11)}).value();
  std::vector<std::uint32_t> positions{7};
  std::vector<std::uint32_t> visible{3};
  DeepSeekRankComputeWorkBuilder builder;
  ASSERT_TRUE(factory.append_decode_plan_work(
      {decode_input(10, positions, visible),
       decode_input(11, positions, visible)}, builder).ok());
  auto work = std::move(builder).finish().value();
  ASSERT_EQ(work.decode_attention.size(), 2U);
  EXPECT_EQ(work.decode_attention[0].work.recent_writer,
            resources(10).recent_writer);
  EXPECT_EQ(work.decode_attention[1].work.attention.query_positions[0], 7U);
  EXPECT_TRUE(work.chunk_attention.empty());
}

TEST(DeepSeekAttentionWorkFactoryTest,
     PublishesChunkBackingAndRejectsCoverageGaps) {
  auto factory = DeepSeekAttentionWorkFactory::Create(
      {10, 11}, {resources(10), resources(11)}).value();
  std::vector<DeepSeekRecentStateSubmission> recent(1);
  recent[0].absolute_position = 9;
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates(1);
  std::vector<std::uint32_t> positions{7};
  std::vector<std::uint32_t> visible{3};
  DeepSeekRankComputeWorkBuilder missing;
  EXPECT_FALSE(factory.append_chunk_plan_work(
      {chunk_input(10, recent, updates, positions, visible)}, missing).ok());
  DeepSeekRankComputeWorkBuilder builder;
  ASSERT_TRUE(factory.append_chunk_plan_work(
      {chunk_input(10, recent, updates, positions, visible),
       chunk_input(11, recent, updates, positions, visible)}, builder).ok());
  auto work = std::move(builder).finish().value();
  ASSERT_EQ(work.chunk_attention.size(), 2U);
  EXPECT_EQ(work.chunk_attention[0].work.submission.recent[0]
                .absolute_position, 9U);
  EXPECT_TRUE(work.decode_attention.empty());
}

TEST(DeepSeekAttentionWorkFactoryTest,
     RejectsIncompleteOrDuplicatedRuntimeResources) {
  auto incomplete = resources(10);
  incomplete.prefill_coordinator = nullptr;
  EXPECT_FALSE(DeepSeekAttentionWorkFactory::Create(
      {10, 11}, {incomplete, resources(11)}).ok());
  EXPECT_FALSE(DeepSeekAttentionWorkFactory::Create(
      {10, 11}, {resources(10), resources(10)}).ok());
}

}  // namespace
}  // namespace pih
