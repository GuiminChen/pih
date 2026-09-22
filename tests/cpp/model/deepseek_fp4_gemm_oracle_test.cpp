#include "pih/model/deepseek_fp4_gemm_oracle.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace pih {
namespace {

TEST(DeepSeekFp4GemmOracleTest, AppliesBothScalesPerK32Subblock) {
  std::vector<float> input(128, 1.0F);
  auto activation = DeepSeekFp8ActivationCodec::Quantize(input, 1, 128);
  ASSERT_TRUE(activation.ok());
  std::vector<std::byte> weights(128, std::byte{0x22});
  std::vector<std::byte> scales(8, std::byte{127});
  // The second output row has weight scale 2 for all four K32 blocks.
  for (std::size_t index = 4; index < 8; ++index) {
    scales[index] = std::byte{128};
  }
  auto result = DeepSeekFp4GemmOracle::Multiply(
      *activation, weights, scales, 2);
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 2U);
  EXPECT_FLOAT_EQ((*result)[0], 128.0F);
  EXPECT_FLOAT_EQ((*result)[1], 256.0F);
}

TEST(DeepSeekFp4GemmOracleTest, PreservesPackedNibbleOrder) {
  std::vector<float> input(128, 0.0F);
  input[0] = 1.0F;
  input[1] = 2.0F;
  auto activation = DeepSeekFp8ActivationCodec::Quantize(input, 1, 128);
  ASSERT_TRUE(activation.ok());
  std::vector<std::byte> weights(64, std::byte{0x00});
  weights[0] = std::byte{0xA2};  // low K is +1, high K is -1
  std::vector<std::byte> scales(4, std::byte{127});
  auto result = DeepSeekFp4GemmOracle::Multiply(
      *activation, weights, scales, 1);
  ASSERT_TRUE(result.ok());
  EXPECT_FLOAT_EQ((*result)[0], -1.0F);
}

TEST(DeepSeekFp4GemmOracleTest, RejectsMalformedEncodedInputs) {
  std::vector<float> input(128, 1.0F);
  auto activation = DeepSeekFp8ActivationCodec::Quantize(input, 1, 128);
  ASSERT_TRUE(activation.ok());
  const std::vector<std::byte> weights(64, std::byte{0x22});
  std::vector<std::byte> scales(4, std::byte{127});
  activation->e4m3_bits[0] = 0x7F;
  EXPECT_FALSE(DeepSeekFp4GemmOracle::Multiply(
                   *activation, weights, scales, 1).ok());
  activation = DeepSeekFp8ActivationCodec::Quantize(input, 1, 128);
  scales[0] = std::byte{0xFF};
  EXPECT_FALSE(DeepSeekFp4GemmOracle::Multiply(
                   *activation, weights, scales, 1).ok());
  EXPECT_FALSE(DeepSeekFp4GemmOracle::Multiply(
                   *activation, weights, std::span<const std::byte>{}, 1).ok());
}

}  // namespace
}  // namespace pih
