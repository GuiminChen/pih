#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"

#include <gtest/gtest.h>

#include <limits>

namespace pih {
namespace {

DeepSeekFp8ActivationQuantLaunch valid_quant_launch() {
  return {.input_bf16 = 1,
          .output_e4m3 = 2,
          .scale_bits = 3,
          .error_flag = 4,
          .stream = 5,
          .token_count = 2,
          .logical_k = 4096};
}

TEST(DeepSeekFp8ActivationLaunchTest, AcceptsPinnedModelProjectionWidths) {
  auto launch = valid_quant_launch();
  EXPECT_TRUE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  launch.logical_k = 2048;
  EXPECT_TRUE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  launch.logical_k = 1024;
  EXPECT_TRUE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  launch.logical_k = 8192;
  EXPECT_TRUE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  launch.logical_k = 12288;
  EXPECT_TRUE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  EXPECT_EQ(DeepSeekFp8ActivationQuantLaunch::kGroupSize, 128U);
}

TEST(DeepSeekFp8ActivationLaunchTest, RejectsMissingOrUnknownShape) {
  auto launch = valid_quant_launch();
  launch.output_e4m3 = 0;
  EXPECT_FALSE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  launch = valid_quant_launch();
  launch.scale_bits = 0;
  EXPECT_FALSE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  launch = valid_quant_launch();
  launch.stream = 0;
  EXPECT_FALSE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  launch = valid_quant_launch();
  launch.token_count = 0;
  EXPECT_FALSE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  launch = valid_quant_launch();
  launch.logical_k = 128;
  EXPECT_FALSE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
  launch = valid_quant_launch();
  launch.token_count = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(validate_deepseek_fp8_activation_quant_launch(launch).ok());
}

}  // namespace
}  // namespace pih
