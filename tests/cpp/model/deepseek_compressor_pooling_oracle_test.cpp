#include "pih/model/deepseek_compressor_pooling_oracle.h"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace pih { namespace {

TEST(DeepSeekCompressorPoolingOracleTest,
     Ratio4CombinesPreviousFirstAndCurrentSecondHalves) {
  auto oracle = DeepSeekCompressorPoolingOracle::Create(4, 1).value();
  const std::vector<float> gate{0.0F, 0.0F};
  const std::vector<float> ape{0.0F, 0.0F};
  std::vector<float> output;
  for (std::uint32_t token = 0; token < 4; ++token) {
    auto result = oracle.step(token,
                              std::vector<float>{10.0F * (token + 1U),
                                                 1.0F + token},
                              gate, ape);
    ASSERT_TRUE(result.ok());
    output = *result;
  }
  ASSERT_EQ(output.size(), 1U);
  EXPECT_FLOAT_EQ(output[0], 2.5F);
  for (std::uint32_t token = 4; token < 8; ++token) {
    auto result = oracle.step(token,
                              std::vector<float>{100.0F + token,
                                                 1.0F + token},
                              gate, ape);
    ASSERT_TRUE(result.ok());
    output = *result;
  }
  ASSERT_EQ(output.size(), 1U);
  EXPECT_FLOAT_EQ(output[0], 15.75F);
  EXPECT_EQ(oracle.completed_slots(), 2U);
  EXPECT_EQ(oracle.remainder_tokens(), 0U);
}

TEST(DeepSeekCompressorPoolingOracleTest, Ratio128PoolsCurrentWindowOnly) {
  auto oracle = DeepSeekCompressorPoolingOracle::Create(128, 1).value();
  const std::vector<float> gate{0.0F};
  const std::vector<float> ape{0.0F};
  std::vector<float> output;
  for (std::uint32_t token = 0; token < 128; ++token) {
    auto result = oracle.step(token, std::vector<float>{1.0F + token},
                              gate, ape);
    ASSERT_TRUE(result.ok());
    output = *result;
  }
  ASSERT_EQ(output.size(), 1U);
  EXPECT_FLOAT_EQ(output[0], 64.5F);
}

TEST(DeepSeekCompressorPoolingOracleTest, SoftmaxIsIndependentPerOutputDim) {
  auto oracle = DeepSeekCompressorPoolingOracle::Create(4, 2).value();
  const std::vector<float> ape(4, 0.0F);
  std::vector<float> output;
  for (std::uint32_t token = 0; token < 4; ++token) {
    const std::vector<float> kv{0.0F, 0.0F, static_cast<float>(token),
                                static_cast<float>(10U + token)};
    const std::vector<float> gate{0.0F, 0.0F,
                                  token == 3 ? 80.0F : 0.0F,
                                  token == 0 ? 80.0F : 0.0F};
    auto result = oracle.step(token, kv, gate, ape);
    ASSERT_TRUE(result.ok());
    output = *result;
  }
  ASSERT_EQ(output.size(), 2U);
  EXPECT_NEAR(output[0], 3.0F, 1e-5F);
  EXPECT_NEAR(output[1], 10.0F, 1e-5F);
}

TEST(DeepSeekCompressorPoolingOracleTest, RejectsGapsAndWrongProjection) {
  auto oracle = DeepSeekCompressorPoolingOracle::Create(4, 1).value();
  const std::vector<float> two(2, 0.0F);
  EXPECT_FALSE(oracle.step(1, two, two, two).ok());
  const std::vector<float> one(1, 0.0F);
  EXPECT_FALSE(oracle.step(0, one, one, one).ok());
}

TEST(DeepSeekCompressorPoolingOracleTest, InvalidInputDoesNotAdvanceState) {
  auto oracle = DeepSeekCompressorPoolingOracle::Create(4, 1).value();
  const std::vector<float> kv{1.0F, 2.0F};
  const std::vector<float> gate{0.0F,
                                std::numeric_limits<float>::infinity()};
  const std::vector<float> ape(2, 0.0F);
  EXPECT_FALSE(oracle.step(0, kv, gate, ape).ok());
  EXPECT_EQ(oracle.remainder_tokens(), 0U);
  const std::vector<float> valid(2, 0.0F);
  EXPECT_TRUE(oracle.step(0, kv, valid, ape).ok());
  EXPECT_EQ(oracle.remainder_tokens(), 1U);
}

} }  // namespace pih
