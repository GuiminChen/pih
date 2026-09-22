#include "pih/backend/cuda/deepseek_fp8_gemm.h"
#include "pih/model/deepseek_fp8_gemm_oracle.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekFp8GemmTest, AppliesK128ActivationAndWeightBlockScales) {
  DeepSeekFp8Activation activation;
  activation.token_count=1; activation.logical_k=256;
  activation.e4m3_bits.assign(256,0x38U);
  activation.scale_bits={127U,127U};
  std::vector<std::byte> weight(128U*256U,std::byte{0x38});
  std::vector<std::byte> scales={std::byte{127},std::byte{128}};
  auto output=DeepSeekFp8GemmOracle::Multiply(
      activation,weight,scales,128);
  ASSERT_TRUE(output.ok()); ASSERT_EQ(output->size(),128U);
  EXPECT_FLOAT_EQ((*output)[0],384.0F);
  EXPECT_FLOAT_EQ((*output)[127],384.0F);
}

TEST(DeepSeekFp8GemmTest, RejectsInvalidScaleAndShape) {
  DeepSeekFp8Activation activation;
  activation.token_count=1; activation.logical_k=128;
  activation.e4m3_bits.assign(128,0x38U); activation.scale_bits={127U};
  std::vector<std::byte> weight(128U*128U,std::byte{0x38});
  std::vector<std::byte> scales={std::byte{0xFF}};
  EXPECT_FALSE(DeepSeekFp8GemmOracle::Multiply(
      activation,weight,scales,128).ok());
  scales[0]=std::byte{127}; weight.pop_back();
  EXPECT_FALSE(DeepSeekFp8GemmOracle::Multiply(
      activation,weight,scales,128).ok());
}

TEST(DeepSeekFp8GemmTest, RejectsInvalidE4m3Operands) {
  DeepSeekFp8Activation activation;
  activation.token_count = 1;
  activation.logical_k = 128;
  activation.e4m3_bits.assign(128, 0x38U);
  activation.scale_bits = {127U};
  std::vector<std::byte> weight(128U * 128U, std::byte{0x38});
  std::vector<std::byte> scales = {std::byte{127}};

  activation.e4m3_bits[0] = 0x7FU;
  EXPECT_FALSE(DeepSeekFp8GemmOracle::Multiply(
                   activation, weight, scales, 128)
                   .ok());
  activation.e4m3_bits[0] = 0x38U;
  weight[0] = std::byte{0xFF};
  EXPECT_FALSE(DeepSeekFp8GemmOracle::Multiply(
                   activation, weight, scales, 128)
                   .ok());
}

TEST(DeepSeekFp8GemmTest, LaunchAcceptsBoundedBlockAlignedGeometry) {
  DeepSeekFp8GemmLaunch launch{1,2,3,4,5,6,7,4096,2048,4096};
  EXPECT_TRUE(validate_deepseek_fp8_gemm_launch(launch).ok());
  launch.n=129280; launch.k=32768;
  EXPECT_TRUE(validate_deepseek_fp8_gemm_launch(launch).ok());
  launch.n=129279;
  EXPECT_FALSE(validate_deepseek_fp8_gemm_launch(launch).ok());
  launch.n=128; launch.k=129;
  EXPECT_FALSE(validate_deepseek_fp8_gemm_launch(launch).ok());
  launch.k=128; launch.m=4097;
  EXPECT_FALSE(validate_deepseek_fp8_gemm_launch(launch).ok());
  launch.m=1; launch.output_type=DeepSeekFp8GemmOutputType::kFp32;
  EXPECT_TRUE(validate_deepseek_fp8_gemm_launch(launch).ok());
  launch.output_type=static_cast<DeepSeekFp8GemmOutputType>(99);
  EXPECT_FALSE(validate_deepseek_fp8_gemm_launch(launch).ok());
}

} }  // namespace pih
