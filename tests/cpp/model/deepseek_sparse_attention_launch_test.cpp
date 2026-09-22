#include "pih/backend/cuda/deepseek_sparse_attention.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekSparseAttentionLaunchTest, AcceptsOfficialMaximumEnvelope) {
  DeepSeekSparseAttentionLaunch launch{
      1, 2, 3, 4, 5, 6, 7, 4096, 64, 1048704, 8320};
  EXPECT_TRUE(validate_deepseek_sparse_attention_launch(launch).ok());
  EXPECT_EQ(DeepSeekSparseAttentionLaunch::kHeadDim, 512U);
  EXPECT_EQ(DeepSeekSparseAttentionLaunch::kTileIndices, 64U);
}

TEST(DeepSeekSparseAttentionLaunchTest, RejectsMissingAndOversizedInputs) {
  DeepSeekSparseAttentionLaunch launch{
      1, 2, 3, 4, 5, 6, 7, 1, 64, 1, 640};
  launch.query_bf16 = 0;
  EXPECT_FALSE(validate_deepseek_sparse_attention_launch(launch).ok());
  launch.query_bf16 = 1;
  launch.index_count = 8321;
  EXPECT_FALSE(validate_deepseek_sparse_attention_launch(launch).ok());
  launch.index_count = 1;
  launch.head_count = 65;
  EXPECT_FALSE(validate_deepseek_sparse_attention_launch(launch).ok());
  launch.head_count = 64;
  launch.query_count = 4097;
  EXPECT_FALSE(validate_deepseek_sparse_attention_launch(launch).ok());
}

TEST(DeepSeekSparseAttentionLaunchTest, ValidatesPagedGeometryAtomically) {
  DeepSeekSparseAttentionLaunch launch{
      1, 2, 3, 4, 5, 6, 7, 1, 64, 256, 1};
  launch.compressed_kv_bf16 = 8;
  launch.page_slots_u32 = 9;
  launch.recent_physical_offset = 0;
  launch.compressed_physical_offset = 128;
  launch.compressed_slot_count = 65;
  launch.logical_page_count = 2;
  launch.physical_page_count = 4;
  EXPECT_TRUE(validate_deepseek_sparse_attention_launch(launch).ok());
  launch.logical_page_count = 1;
  EXPECT_FALSE(validate_deepseek_sparse_attention_launch(launch).ok());
  launch.logical_page_count = 2;
  launch.page_slots_u32 = 0;
  EXPECT_FALSE(validate_deepseek_sparse_attention_launch(launch).ok());
}

} }  // namespace pih
