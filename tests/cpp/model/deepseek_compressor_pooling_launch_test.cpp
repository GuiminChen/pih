#include "pih/backend/cuda/deepseek_compressor_pooling.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekCompressorPoolingLaunchTest, AcceptsOfficialMainAndIndexerShapes) {
  EXPECT_TRUE(validate_deepseek_compressor_pooling_launch(
                  {1, 2, 3, 4, 5, 6, 7, 8, 4096, 4, 512, 1048575})
                  .ok());
  EXPECT_TRUE(validate_deepseek_compressor_pooling_launch(
                  {1, 2, 3, 4, 5, 6, 7, 8, 1, 4, 128, 0})
                  .ok());
  EXPECT_TRUE(validate_deepseek_compressor_pooling_launch(
                  {1, 2, 3, 4, 5, 6, 7, 8, 1, 128, 512, 127})
                  .ok());
}

TEST(DeepSeekCompressorPoolingLaunchTest, RejectsUnsupportedAndMissingInputs) {
  DeepSeekCompressorPoolingLaunch launch{1, 2, 3, 4, 5, 6, 7, 8,
                                         1, 4, 512, 0};
  launch.kv_state_f32 = 0;
  EXPECT_FALSE(validate_deepseek_compressor_pooling_launch(launch).ok());
  launch.kv_state_f32 = 4;
  launch.ratio = 8;
  EXPECT_FALSE(validate_deepseek_compressor_pooling_launch(launch).ok());
  launch.ratio = 4;
  launch.head_dim = 256;
  EXPECT_FALSE(validate_deepseek_compressor_pooling_launch(launch).ok());
  launch.head_dim = 512;
  launch.batch_count = 4097;
  EXPECT_FALSE(validate_deepseek_compressor_pooling_launch(launch).ok());
}

} }  // namespace pih
