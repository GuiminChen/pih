#include "pih/model/deepseek_expert_swiglu_oracle.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace pih {
namespace {

TEST(DeepSeekExpertSwiGluOracleTest, AppliesOfficialAsymmetricClamp) {
  const std::vector<float> gate = {-20.0F, 20.0F, -2.0F};
  const std::vector<float> up = {-20.0F, 20.0F, 3.0F};
  auto result = DeepSeekExpertSwiGluOracle::Apply(gate, up, 1, 3);
  ASSERT_TRUE(result.ok());
  const auto silu = [](float value) {
    return value / (1.0F + std::exp(-value));
  };
  EXPECT_FLOAT_EQ((*result)[0], silu(-20.0F) * -10.0F);
  EXPECT_FLOAT_EQ((*result)[1], silu(10.0F) * 10.0F);
  EXPECT_FLOAT_EQ((*result)[2], silu(-2.0F) * 3.0F);
}

TEST(DeepSeekExpertSwiGluOracleTest, SupportsExplicitCompatibilityLimit) {
  const std::vector<float> gate = {4.0F};
  const std::vector<float> up = {-4.0F};
  auto result = DeepSeekExpertSwiGluOracle::Apply(gate, up, 1, 1, 2.0F);
  ASSERT_TRUE(result.ok());
  EXPECT_FLOAT_EQ((*result)[0],
                  (2.0F / (1.0F + std::exp(-2.0F))) * -2.0F);
}

TEST(DeepSeekExpertSwiGluOracleTest, RejectsShapeAndNonfiniteWithoutOutput) {
  const std::vector<float> one = {1.0F};
  const std::vector<float> two = {1.0F, 2.0F};
  EXPECT_FALSE(DeepSeekExpertSwiGluOracle::Apply(one, one, 0, 1).ok());
  EXPECT_FALSE(DeepSeekExpertSwiGluOracle::Apply(one, two, 1, 1).ok());
  const std::vector<float> invalid = {
      std::numeric_limits<float>::infinity()};
  EXPECT_FALSE(DeepSeekExpertSwiGluOracle::Apply(invalid, one, 1, 1).ok());
}

TEST(DeepSeekExpertSwiGluOracleTest, FreezesFlash0731GeometryAndLimit) {
  EXPECT_EQ(DeepSeekExpertSwiGluOracle::kHiddenSize, 4096U);
  EXPECT_EQ(DeepSeekExpertSwiGluOracle::kIntermediateSize, 2048U);
  EXPECT_FLOAT_EQ(DeepSeekExpertSwiGluOracle::kFlash0731Limit, 10.0F);
}

}  // namespace
}  // namespace pih
