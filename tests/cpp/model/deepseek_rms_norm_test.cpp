#include "pih/backend/cuda/deepseek_rms_norm.h"
#include "pih/model/deepseek_rms_norm_oracle.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekRmsNormTest, OracleUsesPinnedEpsilonAndBf16Boundary) {
  std::vector<BFloat16> input(512,BFloat16::FromFloat(2.0F));
  std::vector<BFloat16> weight(512,BFloat16::FromFloat(0.5F));
  std::vector<BFloat16> output(512);
  ASSERT_TRUE(deepseek_rms_norm_oracle(input,weight,1,512,output).ok());
  const auto expected=1.0F/std::sqrt(4.0F+0.000001F);
  EXPECT_FLOAT_EQ(output[0].to_float(),
                  BFloat16::FromFloat(expected).to_float());
}

TEST(DeepSeekRmsNormTest, OracleRejectsWrongShapeAndDoesNotPublish) {
  std::vector<BFloat16> input(512,BFloat16::FromFloat(1.0F));
  std::vector<BFloat16> weight(512,BFloat16::FromFloat(1.0F));
  std::vector<BFloat16> output(512,BFloat16::FromFloat(7.0F));
  input[3]=BFloat16::FromFloat(std::nanf(""));
  EXPECT_FALSE(deepseek_rms_norm_oracle(input,weight,1,512,output).ok());
  EXPECT_FLOAT_EQ(output[0].to_float(),7.0F);
  EXPECT_FALSE(deepseek_rms_norm_oracle(input,weight,1,256,output).ok());
}

TEST(DeepSeekRmsNormTest, LaunchAcceptsOnlyV1ShapesAndEpsilon) {
  DeepSeekRmsNormLaunch launch{1,2,3,4,5,8,4096,
                               DeepSeekRmsNormLaunch::kEpsilon};
  EXPECT_TRUE(validate_deepseek_rms_norm_launch(launch).ok());
  for (const auto hidden : {512U,1024U,4096U}) {
    launch.hidden_size=hidden;
    EXPECT_TRUE(validate_deepseek_rms_norm_launch(launch).ok());
  }
  launch.hidden_size=2048;
  EXPECT_FALSE(validate_deepseek_rms_norm_launch(launch).ok());
  launch.hidden_size=4096; launch.epsilon=0.00001F;
  EXPECT_FALSE(validate_deepseek_rms_norm_launch(launch).ok());
  launch.epsilon=DeepSeekRmsNormLaunch::kEpsilon; launch.error_flag=0;
  EXPECT_FALSE(validate_deepseek_rms_norm_launch(launch).ok());
}

TEST(DeepSeekRmsNormTest, FusedResidualLaunchPreservesExactV1Contract) {
  DeepSeekFusedResidualRmsNormLaunch launch{
      1, 2, 3, 4, 5, 6, 7, 8, 4096,
      DeepSeekFusedResidualRmsNormLaunch::kEpsilon};
  EXPECT_TRUE(validate_deepseek_fused_residual_rms_norm_launch(launch).ok());
  launch.residual_output_bf16 = launch.input_bf16;
  EXPECT_FALSE(validate_deepseek_fused_residual_rms_norm_launch(launch).ok());
  launch.residual_output_bf16 = 4;
  launch.hidden_size = 2048;
  EXPECT_FALSE(validate_deepseek_fused_residual_rms_norm_launch(launch).ok());
  launch.hidden_size = 4096;
  launch.epsilon = 1.0e-5F;
  EXPECT_FALSE(validate_deepseek_fused_residual_rms_norm_launch(launch).ok());
}

} }  // namespace pih
