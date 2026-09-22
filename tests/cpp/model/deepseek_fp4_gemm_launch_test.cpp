#include "pih/backend/cuda/deepseek_fp4_gemm.h"

#include <gtest/gtest.h>

#include <limits>

namespace pih {
namespace {

DeepSeekFp4GemmLaunch valid_w1() {
  return {.activation_e4m3 = 1,
          .activation_scale_bits = 2,
          .packed_weight = 3,
          .weight_scale_bits = 4,
          .output_bf16 = 5,
          .error_flag = 6,
          .stream = 7,
          .m = 2,
          .n = 2048,
          .k = 4096};
}

TEST(DeepSeekFp4GemmLaunchTest, AcceptsW1W3AndW2Shapes) {
  auto launch = valid_w1();
  EXPECT_TRUE(validate_deepseek_fp4_gemm_launch(launch).ok());
  launch.n = 4096;
  launch.k = 2048;
  EXPECT_TRUE(validate_deepseek_fp4_gemm_launch(launch).ok());
}

TEST(DeepSeekFp4GemmLaunchTest, RejectsMissingAndInventedShapes) {
  auto launch = valid_w1();
  launch.packed_weight = 0;
  EXPECT_FALSE(validate_deepseek_fp4_gemm_launch(launch).ok());
  launch = valid_w1();
  launch.activation_scale_bits = 0;
  EXPECT_FALSE(validate_deepseek_fp4_gemm_launch(launch).ok());
  launch = valid_w1();
  launch.stream = 0;
  EXPECT_FALSE(validate_deepseek_fp4_gemm_launch(launch).ok());
  launch = valid_w1();
  launch.n = 4096;
  EXPECT_FALSE(validate_deepseek_fp4_gemm_launch(launch).ok());
  launch = valid_w1();
  launch.m = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(validate_deepseek_fp4_gemm_launch(launch).ok());
}

}  // namespace
}  // namespace pih
