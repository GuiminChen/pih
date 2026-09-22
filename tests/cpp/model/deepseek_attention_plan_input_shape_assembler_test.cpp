#include "pih/model/deepseek_attention_plan_input_shape_assembler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekAttentionLayerPlanShapeSeed seed(
    std::uint32_t ratio, std::vector<std::uint32_t> positions) {
  DeepSeekAttentionLayerPlanShapeSeed result;
  result.layer = 6;
  result.ratio = ratio;
  result.positions = positions;
  result.index_query_bf16 = 1;
  result.index_kv_bf16 = 2;
  result.index_head_weight_f32 = 3;
  result.index_arena = {4, 5};
  result.sparse_query_bf16 = 6;
  result.sparse_kv_bf16 = 7;
  result.attention_sink_f32 = 8;
  result.sparse_indices_i32 = 9;
  result.sparse_page_slots_u32 = 13;
  result.sparse_output_bf16 = 10;
  result.sparse_error_u32 = 11;
  result.sparse_kv_count = 4096;
  result.recent_physical_offset = 0;
  result.compressed_physical_offset = 1024;
  result.stream = 12;
  if (ratio == 4 && (positions.back() + 1U) / 4U != 0) {
    result.has_indexer_projection = true;
    result.indexer_projection =
        {13, 14, 15, 23, 24, 25, 16, 17, 18, 1, 3, 19, 12,
         static_cast<std::uint32_t>(positions.size()), 1048576};
  }
  for (const auto position : positions) {
    result.recent.push_back({6, 20, 12, 1, position});
    DeepSeekCompressedLayerUpdateSubmission update;
    update.ratio = ratio;
    update.main_state.layer_id = 6;
    update.main_state.absolute_position = position;
    update.main_state.stream = 12;
    result.updates.push_back(update);
  }
  return result;
}

TEST(DeepSeekAttentionPlanInputShapeAssemblerTest,
     DerivesRatio4DecodeVisibilityAndCompressedCoverage) {
  auto owned = DeepSeekOwnedAttentionLayerPlanInput::AssembleDecode(
      seed(4, {7}));
  ASSERT_TRUE(owned.ok()) << owned.status().message();
  ASSERT_TRUE(owned->is_decode());
  const auto& input = owned->decode();
  EXPECT_EQ(input.layer, 6U);
  EXPECT_EQ(input.attention.kind,
            DeepSeekCompressedAttentionKind::kRatio4);
  EXPECT_EQ(input.attention.compressed_slot_count, 2U);
  EXPECT_EQ(input.attention.query_positions[0], 7U);
  EXPECT_EQ(input.attention.selection.visible_slot_counts[0], 2U);
  EXPECT_EQ(input.attention.selection.query_count, 1U);
  EXPECT_EQ(input.attention.selection.head_count, 64U);
  EXPECT_EQ(input.attention.attention.kv_count, 4096U);
}

TEST(DeepSeekAttentionPlanInputShapeAssemblerTest,
     Ratio128ChunkOwnsViewsAcrossMove) {
  auto logical = seed(128, {127, 128});
  logical.index_query_bf16 = 0;
  logical.index_kv_bf16 = 0;
  logical.index_head_weight_f32 = 0;
  logical.index_arena = {};
  auto result = DeepSeekOwnedAttentionLayerPlanInput::AssembleChunk(
      std::move(logical));
  ASSERT_TRUE(result.ok()) << result.status().message();
  auto owned = std::move(*result);
  ASSERT_FALSE(owned.is_decode());
  const auto& input = owned.chunk();
  EXPECT_EQ(input.submission.recent.size(), 2U);
  EXPECT_EQ(input.submission.updates.size(), 2U);
  EXPECT_EQ(input.submission.attention.query_positions[0], 127U);
  EXPECT_EQ(input.submission.attention.query_positions[1], 128U);
  EXPECT_EQ(input.submission.attention.compressed_slot_count, 1U);
  EXPECT_TRUE(
      input.submission.attention.selection.visible_slot_counts.empty());
}

TEST(DeepSeekAttentionPlanInputShapeAssemblerTest,
     RejectsNonContiguousOrMismatchedSchedulerState) {
  EXPECT_FALSE(DeepSeekOwnedAttentionLayerPlanInput::AssembleChunk(
      seed(4, {3, 5})).ok());
  auto mismatched = seed(4, {3});
  mismatched.recent[0].absolute_position = 2;
  EXPECT_FALSE(DeepSeekOwnedAttentionLayerPlanInput::AssembleDecode(
      std::move(mismatched)).ok());
}

TEST(DeepSeekAttentionPlanInputShapeAssemblerTest,
     PropagatesDeferredRatio4PageResourcesForDecode) {
  auto input = seed(4, {7});
  input.bind_compressed_page_mutations = true;
  input.main_rms_weight_bf16 = 21;
  input.index_rms_weight_bf16 = 22;
  input.main_cos_sin_cache_f32 = 23;
  input.index_cos_sin_cache_f32 = 24;
  input.rms_epsilon = 1.0e-6F;
  auto owned = DeepSeekOwnedAttentionLayerPlanInput::AssembleDecode(
      std::move(input));
  ASSERT_TRUE(owned.ok()) << owned.status().message();
  const auto& update = owned->decode().update;
  EXPECT_TRUE(update.defer_page_mutation);
  EXPECT_EQ(update.deferred_page.main_rms_weight_bf16, 21U);
  EXPECT_EQ(update.deferred_page.index_rms_weight_bf16, 22U);
  EXPECT_EQ(update.deferred_page.main_cos_sin_cache_f32, 23U);
  EXPECT_EQ(update.deferred_page.index_cos_sin_cache_f32, 24U);
  EXPECT_FLOAT_EQ(update.deferred_page.rms_epsilon, 1.0e-6F);
}

TEST(DeepSeekAttentionPlanInputShapeAssemblerTest,
     RejectsInvalidDeferredPageResources) {
  auto missing = seed(4, {7});
  missing.bind_compressed_page_mutations = true;
  EXPECT_FALSE(DeepSeekOwnedAttentionLayerPlanInput::AssembleDecode(
      std::move(missing)).ok());

  auto ratio128_with_index = seed(128, {127});
  ratio128_with_index.bind_compressed_page_mutations = true;
  ratio128_with_index.main_rms_weight_bf16 = 21;
  ratio128_with_index.main_cos_sin_cache_f32 = 23;
  ratio128_with_index.index_rms_weight_bf16 = 22;
  ratio128_with_index.rms_epsilon = 1.0e-6F;
  EXPECT_FALSE(DeepSeekOwnedAttentionLayerPlanInput::AssembleDecode(
      std::move(ratio128_with_index)).ok());

}

TEST(DeepSeekAttentionPlanInputShapeAssemblerTest,
     PropagatesDeferredPageResourcesAcrossChunk) {
  auto chunk = seed(4, {7, 8});
  chunk.bind_compressed_page_mutations = true;
  chunk.main_rms_weight_bf16 = 21;
  chunk.index_rms_weight_bf16 = 22;
  chunk.main_cos_sin_cache_f32 = 23;
  chunk.index_cos_sin_cache_f32 = 24;
  chunk.rms_epsilon = 1.0e-6F;
  auto owned = DeepSeekOwnedAttentionLayerPlanInput::AssembleChunk(
      std::move(chunk));
  ASSERT_TRUE(owned.ok()) << owned.status().message();
  ASSERT_EQ(owned->chunk().submission.updates.size(), 2U);
  for (const auto& update : owned->chunk().submission.updates) {
    EXPECT_TRUE(update.defer_page_mutation);
    EXPECT_EQ(update.deferred_page.main_rms_weight_bf16, 21U);
    EXPECT_EQ(update.deferred_page.index_rms_weight_bf16, 22U);
    EXPECT_FLOAT_EQ(update.deferred_page.rms_epsilon, 1.0e-6F);
  }
}

}  // namespace
}  // namespace pih
