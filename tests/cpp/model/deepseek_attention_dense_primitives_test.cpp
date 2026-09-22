#include "pih/backend/cuda/deepseek_attention_dense_primitives.h"
#include "pih/model/deepseek_attention_dense_oracle.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace pih { namespace {

TEST(DeepSeekAttentionDensePrimitivesTest, NormalizesEachHeadIndependently) {
  std::vector<BFloat16> input{
      BFloat16::FromFloat(3), BFloat16::FromFloat(4),
      BFloat16::FromFloat(0), BFloat16::FromFloat(2)};
  std::vector<BFloat16> output(4);
  ASSERT_TRUE(deepseek_head_rms_oracle(input, 1, 2, 2, 1.0e-6F, output).ok());
  EXPECT_EQ(output[0], BFloat16::FromFloat(
      3.0F / std::sqrt(12.5F + 1.0e-6F)));
  EXPECT_EQ(output[2], BFloat16::FromFloat(0.0F));
  EXPECT_EQ(output[3], BFloat16::FromFloat(
      2.0F / std::sqrt(2.0F + 1.0e-6F)));
}

TEST(DeepSeekAttentionDensePrimitivesTest,
     RotaryUsesAdjacentComplexPairsAndSupportsInverse) {
  std::vector<BFloat16> input{
      BFloat16::FromFloat(9), BFloat16::FromFloat(8),
      BFloat16::FromFloat(1), BFloat16::FromFloat(2)};
  const std::vector<float> frequencies{0.0F, 1.0F};
  std::vector<BFloat16> rotated(4), restored(4);
  ASSERT_TRUE(deepseek_rotary_oracle(input, frequencies, 1, 1, 4, 2,
                                     false, rotated).ok());
  EXPECT_EQ(rotated[0], input[0]);
  EXPECT_EQ(rotated[1], input[1]);
  EXPECT_EQ(rotated[2], BFloat16::FromFloat(-2.0F));
  EXPECT_EQ(rotated[3], BFloat16::FromFloat(1.0F));
  ASSERT_TRUE(deepseek_rotary_oracle(rotated, frequencies, 1, 1, 4, 2,
                                     true, restored).ok());
  EXPECT_EQ(restored, input);
}

TEST(DeepSeekAttentionDensePrimitivesTest, LaunchesPinV4AttentionGeometry) {
  DeepSeekHeadRmsLaunch rms{1, 2, 3, 4, 2, 64, 512, 1.0e-6F};
  EXPECT_TRUE(validate_deepseek_head_rms_launch(rms).ok());
  rms.head_count = 63;
  EXPECT_FALSE(validate_deepseek_head_rms_launch(rms).ok());

  DeepSeekRotaryLaunch rotary{1, 2, 3, 4, 2, 64, 512, 64, false, 5, 104};
  EXPECT_TRUE(validate_deepseek_rotary_launch(rotary).ok());
  rotary.inverse = true;
  EXPECT_TRUE(validate_deepseek_rotary_launch(rotary).ok());
  rotary.rope_dimension = 62;
  EXPECT_FALSE(validate_deepseek_rotary_launch(rotary).ok());

  DeepSeekKvFp8SimulateLaunch simulate{1, 2, 3, 2, 512, 448, 64};
  EXPECT_TRUE(validate_deepseek_kv_fp8_simulate_launch(simulate).ok());
  simulate.group_size = 128;
  EXPECT_FALSE(validate_deepseek_kv_fp8_simulate_launch(simulate).ok());
}

TEST(DeepSeekAttentionDensePrimitivesTest,
     KvFp8SimulationPreservesTheRopeTailExactly) {
  std::vector<BFloat16> input(512, BFloat16::FromFloat(0.3F));
  for (std::uint32_t column = 448; column < 512; ++column)
    input[column] = BFloat16::FromFloat(static_cast<float>(column));
  std::vector<BFloat16> output(512);
  ASSERT_TRUE(deepseek_kv_fp8_simulate_oracle(input, 1, 512, 448, 64,
                                               output).ok());
  for (std::uint32_t column = 448; column < 512; ++column)
    EXPECT_EQ(output[column], input[column]);
  EXPECT_NE(output[0], input[0]);
}

TEST(DeepSeekAttentionDensePrimitivesTest,
     GroupedWoAUsesIndependentWeightsAndConcatenatesGroups) {
  const std::vector<BFloat16> input{
      BFloat16::FromFloat(1), BFloat16::FromFloat(2),
      BFloat16::FromFloat(3), BFloat16::FromFloat(4)};
  const std::vector<BFloat16> weight{
      BFloat16::FromFloat(1), BFloat16::FromFloat(0),
      BFloat16::FromFloat(0), BFloat16::FromFloat(1),
      BFloat16::FromFloat(2), BFloat16::FromFloat(0),
      BFloat16::FromFloat(0), BFloat16::FromFloat(2)};
  std::vector<BFloat16> output(4);
  ASSERT_TRUE(deepseek_grouped_bf16_gemm_oracle(
      input, weight, 1, 2, 2, 2, output).ok());
  EXPECT_FLOAT_EQ(output[0].to_float(), 1.0F);
  EXPECT_FLOAT_EQ(output[1].to_float(), 2.0F);
  EXPECT_FLOAT_EQ(output[2].to_float(), 6.0F);
  EXPECT_FLOAT_EQ(output[3].to_float(), 8.0F);
}

TEST(DeepSeekAttentionDensePrimitivesTest, GroupedWoALaunchPinsV4Geometry) {
  DeepSeekGroupedBf16GemmLaunch launch{1, 2, 3, 4, 5, 2, 8, 1024, 4096};
  EXPECT_TRUE(validate_deepseek_grouped_bf16_gemm_launch(launch).ok());
  launch.group_count = 4;
  EXPECT_FALSE(validate_deepseek_grouped_bf16_gemm_launch(launch).ok());
}

TEST(DeepSeekAttentionDensePrimitivesTest,
     ValidatesCheckpointNativeGroupedFp8Projection) {
  DeepSeekGroupedFp8GemmLaunch launch{
      1, 2, 3, 4, 5, 6, 7, 8, 2, 8, 1024, 4096};
  EXPECT_TRUE(validate_deepseek_grouped_fp8_gemm_launch(launch).ok());
  launch.weight_scale_bits = 0;
  EXPECT_FALSE(validate_deepseek_grouped_fp8_gemm_launch(launch).ok());
}

}}  // namespace pih::<anonymous>
