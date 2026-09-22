#include "pih/backend/cuda/deepseek_router_bf16_gemm.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(DeepSeekRouterBf16GemmLaunchTest, AcceptsCheckpointRouterContract) {
  const DeepSeekRouterBf16GemmLaunch launch{
      0x10000, 0x20000, 0x30000, 0x40000, 0x50000, 7, 256, 4096};
  EXPECT_TRUE(validate_deepseek_router_bf16_gemm_launch(launch).ok());
}

TEST(DeepSeekRouterBf16GemmLaunchTest, RejectsWrongShapeAndAliasing) {
  DeepSeekRouterBf16GemmLaunch launch{
      0x10000, 0x20000, 0x30000, 0x40000, 0x50000, 7, 256, 4096};
  launch.expert_count = 255;
  EXPECT_FALSE(validate_deepseek_router_bf16_gemm_launch(launch).ok());
  launch.expert_count = 256;
  launch.scores_f32 = launch.input_bf16;
  EXPECT_FALSE(validate_deepseek_router_bf16_gemm_launch(launch).ok());
}

}  // namespace
}  // namespace pih
