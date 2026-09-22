#include "pih/backend/cuda/deepseek_compressor_bf16_projection.h"

#include <gtest/gtest.h>

namespace pih { namespace {

DeepSeekCompressorBf16ProjectionLaunch valid(std::uint32_t ratio,
                                             std::uint32_t head_dim) {
  return {0x10000, 0x20000, 0x28000, 0x30000, 0x40000, 0x50000, 0x60000,
          7, ratio, head_dim, 4096};
}

TEST(DeepSeekCompressorBf16ProjectionLaunchTest,
     AcceptsMainAndIndexerCheckpointGeometries) {
  EXPECT_TRUE(validate_deepseek_compressor_bf16_projection_launch(
                  valid(4, 512)).ok());
  EXPECT_TRUE(validate_deepseek_compressor_bf16_projection_launch(
                  valid(4, 128)).ok());
  EXPECT_TRUE(validate_deepseek_compressor_bf16_projection_launch(
                  valid(128, 512)).ok());
}

TEST(DeepSeekCompressorBf16ProjectionLaunchTest,
     RejectsForeignGeometryAndOutputAliasing) {
  EXPECT_FALSE(validate_deepseek_compressor_bf16_projection_launch(
                   valid(128, 128)).ok());
  auto launch = valid(4, 512);
  launch.gate_projection_f32 = launch.kv_projection_f32;
  EXPECT_FALSE(validate_deepseek_compressor_bf16_projection_launch(launch)
                   .ok());
  launch = valid(4, 512);
  launch.gate_weight_bf16 = 0;
  EXPECT_FALSE(validate_deepseek_compressor_bf16_projection_launch(launch).ok());
  launch = valid(4, 512);
  launch.token_count = 4097;
  EXPECT_FALSE(validate_deepseek_compressor_bf16_projection_launch(launch)
                   .ok());
}

} }  // namespace pih
