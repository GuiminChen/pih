#include "pih/backend/cuda/deepseek_compressor_bf16_store.h"

#include <gtest/gtest.h>

#include <limits>

namespace pih { namespace {

DeepSeekCompressorBf16StoreLaunch valid(std::uint32_t head_dim) {
  return {1, 2, 3, 4, 5, 6, head_dim, 64, 508, 1.0e-6F};
}

TEST(DeepSeekCompressorBf16StoreLaunchTest,
     AcceptsMainAndIndexerBf16Profiles) {
  EXPECT_TRUE(validate_deepseek_compressor_bf16_store_launch(valid(512)).ok());
  EXPECT_TRUE(validate_deepseek_compressor_bf16_store_launch(valid(128)).ok());
}

TEST(DeepSeekCompressorBf16StoreLaunchTest,
     RejectsMalformedGeometryPointersAndNumerics) {
  auto launch = valid(512);
  launch.destination_bf16 = 0;
  EXPECT_FALSE(validate_deepseek_compressor_bf16_store_launch(launch).ok());
  launch = valid(512);
  launch.rope_head_dim = 32;
  EXPECT_FALSE(validate_deepseek_compressor_bf16_store_launch(launch).ok());
  launch = valid(256);
  EXPECT_FALSE(validate_deepseek_compressor_bf16_store_launch(launch).ok());
  launch = valid(512);
  launch.rope_position = 1048576;
  EXPECT_FALSE(validate_deepseek_compressor_bf16_store_launch(launch).ok());
  launch = valid(512);
  launch.rms_epsilon = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(validate_deepseek_compressor_bf16_store_launch(launch).ok());
}

} }  // namespace pih
