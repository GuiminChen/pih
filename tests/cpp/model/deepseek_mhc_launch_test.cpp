#include "pih/backend/cuda/deepseek_mhc.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekMhcLaunchTest, AcceptsOnlyPinnedPreGeometry) {
  DeepSeekMhcPreLaunch launch{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4096,
                              1.0e-6F, 1.0e-6F, 1.0e-6F, 2.0F, 20};
  EXPECT_TRUE(validate_deepseek_mhc_pre_launch(launch).ok());
  launch.hidden_size = 2048;
  EXPECT_FALSE(validate_deepseek_mhc_pre_launch(launch).ok());
  launch.hidden_size = 4096;
  launch.post_multiplier = 1.0F;
  EXPECT_FALSE(validate_deepseek_mhc_pre_launch(launch).ok());
  launch.post_multiplier = 2.0F;
  launch.sinkhorn_iterations = 8;
  EXPECT_FALSE(validate_deepseek_mhc_pre_launch(launch).ok());
  launch.sinkhorn_iterations = 20;
  launch.norm_weight_bf16 = 0;
  EXPECT_FALSE(validate_deepseek_mhc_pre_launch(launch).ok());
  launch.norm_weight_bf16 = 5;
  launch.layer_input_bf16 = launch.residual_bf16;
  EXPECT_FALSE(validate_deepseek_mhc_pre_launch(launch).ok());
}

TEST(DeepSeekMhcLaunchTest, RejectsInPlacePostExpansion) {
  DeepSeekMhcPostLaunch launch{1, 2, 3, 4, 5, 6, 7, 2, 4096};
  EXPECT_TRUE(validate_deepseek_mhc_post_launch(launch).ok());
  launch.output_bf16 = launch.residual_bf16;
  EXPECT_FALSE(validate_deepseek_mhc_post_launch(launch).ok());
  launch.output_bf16 = 5;
  launch.token_count = 4097;
  EXPECT_FALSE(validate_deepseek_mhc_post_launch(launch).ok());
}

TEST(DeepSeekMhcLaunchTest, AcceptsOnlyPinnedTargetHiddenTapGeometry) {
  DeepSeekMhcTargetHiddenTapLaunch launch{
      1, 2, 3, 4, 5, 4096, 4, 3, 2};
  EXPECT_TRUE(validate_deepseek_mhc_target_hidden_tap_launch(launch).ok());
  launch.target_stage_index = 3;
  EXPECT_FALSE(validate_deepseek_mhc_target_hidden_tap_launch(launch).ok());
  launch.target_stage_index = 2;
  launch.source_stream_count = 1;
  EXPECT_FALSE(validate_deepseek_mhc_target_hidden_tap_launch(launch).ok());
  launch.source_stream_count = 4;
  launch.residual_hc_bf16 = launch.target_hidden_bf16;
  EXPECT_FALSE(validate_deepseek_mhc_target_hidden_tap_launch(launch).ok());
}

} }  // namespace pih
