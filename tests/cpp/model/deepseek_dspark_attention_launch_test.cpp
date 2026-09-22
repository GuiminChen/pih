#include "pih/backend/cuda/deepseek_dspark_attention.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekDsparkAttentionLaunch valid_launch() {
  return {1, 2, 3, 4, 5, 6, 7,
          DeepSeekDsparkAttentionLaunch::kBlockSize, 128};
}

TEST(DeepSeekDsparkAttentionLaunchTest, FreezesOfficialDecodeGeometry) {
  const auto launch = valid_launch();
  EXPECT_TRUE(validate_deepseek_dspark_attention_launch(launch).ok());
  EXPECT_EQ(DeepSeekDsparkAttentionLaunch::kBlockSize, 5U);
  EXPECT_EQ(DeepSeekDsparkAttentionLaunch::kHeadCount, 64U);
  EXPECT_EQ(DeepSeekDsparkAttentionLaunch::kHeadDimension, 512U);
  EXPECT_EQ(DeepSeekDsparkAttentionLaunch::kWindowSize, 128U);
}

TEST(DeepSeekDsparkAttentionLaunchTest, AcceptsPartialRecentWindow) {
  auto launch = valid_launch();
  launch.recent_count = 1;
  EXPECT_TRUE(validate_deepseek_dspark_attention_launch(launch).ok());
  launch.recent_count = 127;
  EXPECT_TRUE(validate_deepseek_dspark_attention_launch(launch).ok());
}

TEST(DeepSeekDsparkAttentionLaunchTest, RejectsGeometryAndMissingResources) {
  auto launch = valid_launch();
  launch.query_count = 4;
  EXPECT_FALSE(validate_deepseek_dspark_attention_launch(launch).ok());
  launch = valid_launch();
  launch.recent_count = 0;
  EXPECT_FALSE(validate_deepseek_dspark_attention_launch(launch).ok());
  launch.recent_count = 129;
  EXPECT_FALSE(validate_deepseek_dspark_attention_launch(launch).ok());
  launch = valid_launch();
  launch.draft_kv_bf16 = 0;
  EXPECT_FALSE(validate_deepseek_dspark_attention_launch(launch).ok());
}

TEST(DeepSeekDsparkAttentionLaunchTest, ValidatesFuturePositionEnvelope) {
  DeepSeekDsparkPositionLaunch launch{1, 2, 3, 127, 133};
  EXPECT_TRUE(validate_deepseek_dspark_position_launch(launch).ok());
  launch.table_position_count = 132;
  EXPECT_FALSE(validate_deepseek_dspark_position_launch(launch).ok());
  launch = {1, 2, 3, UINT32_MAX - 4U, UINT32_MAX};
  EXPECT_FALSE(validate_deepseek_dspark_position_launch(launch).ok());
  launch = {0, 2, 3, 0, 6};
  EXPECT_FALSE(validate_deepseek_dspark_position_launch(launch).ok());
}

}  // namespace
}  // namespace pih
