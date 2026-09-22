#include "pih/backend/cuda/deepseek_expert_swiglu.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekExpertSwiGluLaunch valid_launch() {
  return DeepSeekExpertSwiGluLaunch{
      .gate_bf16 = 1,
      .up_bf16 = 2,
      .route_weights_f32 = 3,
      .output_bf16 = 4,
      .error_flag = 5,
      .stream = 6,
      .token_count = 2,
  };
}

TEST(DeepSeekExpertSwiGluLaunchTest, AcceptsCompleteProductionLaunch) {
  EXPECT_TRUE(validate_deepseek_expert_swiglu_launch(valid_launch()).ok());
  EXPECT_EQ(DeepSeekExpertSwiGluLaunch::kIntermediateSize, 2048U);
  EXPECT_FLOAT_EQ(DeepSeekExpertSwiGluLaunch::kLimit, 10.0F);
}

TEST(DeepSeekExpertSwiGluLaunchTest, RejectsEveryMissingResource) {
  auto launch = valid_launch();
  launch.gate_bf16 = 0;
  EXPECT_FALSE(validate_deepseek_expert_swiglu_launch(launch).ok());
  launch = valid_launch();
  launch.route_weights_f32 = 0;
  EXPECT_FALSE(validate_deepseek_expert_swiglu_launch(launch).ok());
  launch = valid_launch();
  launch.error_flag = 0;
  EXPECT_FALSE(validate_deepseek_expert_swiglu_launch(launch).ok());
  launch = valid_launch();
  launch.stream = 0;
  EXPECT_FALSE(validate_deepseek_expert_swiglu_launch(launch).ok());
  launch = valid_launch();
  launch.token_count = 0;
  EXPECT_FALSE(validate_deepseek_expert_swiglu_launch(launch).ok());
}

TEST(DeepSeekExpertSwiGluLaunchTest,
     SharedExpertDoesNotRequireRouteWeights) {
  DeepSeekSharedExpertSwiGluLaunch launch{1, 2, 3, 4, 5, 4096};
  EXPECT_TRUE(validate_deepseek_shared_expert_swiglu_launch(launch).ok());
  EXPECT_EQ(DeepSeekSharedExpertSwiGluLaunch::kIntermediateSize, 2048U);
  EXPECT_FLOAT_EQ(DeepSeekSharedExpertSwiGluLaunch::kLimit, 10.0F);
  launch.output_bf16 = 0;
  EXPECT_FALSE(validate_deepseek_shared_expert_swiglu_launch(launch).ok());
  launch.output_bf16 = 3;
  launch.token_count = 4097;
  EXPECT_FALSE(validate_deepseek_shared_expert_swiglu_launch(launch).ok());
}

}  // namespace
}  // namespace pih
