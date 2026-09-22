#include "pih/backend/cuda/deepseek_index_score.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekIndexScoreLaunchTest, AcceptsBoundedOfficialShapeTile) {
  const DeepSeekIndexScoreLaunch launch{1, 2, 3, 4, 5, 6,
                                        4096, 64, 4096};
  EXPECT_TRUE(validate_deepseek_index_score_launch(launch).ok());
  EXPECT_EQ(DeepSeekIndexScoreLaunch::kHeadDim, 128U);
  EXPECT_EQ(DeepSeekIndexScoreLaunch::kMaximumHeads, 64U);
  EXPECT_EQ(DeepSeekIndexScoreLaunch::kMaximumSlotTile, 4096U);
}

TEST(DeepSeekIndexScoreLaunchTest, RejectsMissingAndUnboundedWorkspace) {
  DeepSeekIndexScoreLaunch launch{1, 2, 3, 4, 5, 6, 1, 64, 4096};
  launch.query_bf16 = 0;
  EXPECT_FALSE(validate_deepseek_index_score_launch(launch).ok());
  launch.query_bf16 = 1;
  launch.slot_count = 4097;
  EXPECT_FALSE(validate_deepseek_index_score_launch(launch).ok());
  launch.slot_count = 1;
  launch.head_count = 65;
  EXPECT_FALSE(validate_deepseek_index_score_launch(launch).ok());
  launch.head_count = 64;
  launch.query_count = 4097;
  EXPECT_FALSE(validate_deepseek_index_score_launch(launch).ok());
}

TEST(DeepSeekIndexScoreLaunchTest, ValidatesPagedTileCoverage) {
  DeepSeekIndexScoreLaunch launch{1, 2, 3, 4, 5, 6, 1, 64, 65,
                                  7, 63, 2, 3};
  EXPECT_TRUE(validate_deepseek_index_score_launch(launch).ok());
  launch.slot_count = 66;
  EXPECT_FALSE(validate_deepseek_index_score_launch(launch).ok());
  launch.slot_count = 65;
  launch.physical_page_count = 0;
  EXPECT_FALSE(validate_deepseek_index_score_launch(launch).ok());
  launch.physical_page_count = 3;
  launch.page_slots_u32 = 0;
  EXPECT_FALSE(validate_deepseek_index_score_launch(launch).ok());
}

} }  // namespace pih
