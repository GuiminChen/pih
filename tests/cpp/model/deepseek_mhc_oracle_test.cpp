#include "pih/model/deepseek_mhc_oracle.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace pih { namespace {

struct Fixture final {
  static constexpr std::uint32_t kHidden = 2;
  std::vector<BFloat16> residual{
      BFloat16::FromFloat(1), BFloat16::FromFloat(1),
      BFloat16::FromFloat(2), BFloat16::FromFloat(2),
      BFloat16::FromFloat(3), BFloat16::FromFloat(3),
      BFloat16::FromFloat(4), BFloat16::FromFloat(4)};
  std::vector<float> fn = std::vector<float>(24U * 4U * kHidden, 0.0F);
  std::vector<float> scale{1, 1, 1};
  std::vector<float> base = std::vector<float>(24, 0.0F);
  std::vector<float> post = std::vector<float>(4);
  std::vector<float> mix = std::vector<float>(16);
  std::vector<BFloat16> input = std::vector<BFloat16>(kHidden);
  DeepSeekMhcParameters parameters() const {
    return {fn, scale, base, 1.0e-6F, 0.0F, 0.0F, 2.0F, 1};
  }
};

TEST(DeepSeekMhcOracleTest, ReproducesUniformPreAndPostMixing) {
  Fixture fixture;
  ASSERT_TRUE(deepseek_mhc_pre_oracle(
      fixture.residual, fixture.parameters(), 1, Fixture::kHidden,
      fixture.post, fixture.mix, fixture.input).ok());
  for (const auto value : fixture.post) EXPECT_FLOAT_EQ(value, 1.0F);
  for (const auto value : fixture.mix) EXPECT_FLOAT_EQ(value, 0.25F);
  for (const auto value : fixture.input) EXPECT_FLOAT_EQ(value.to_float(), 5.0F);

  std::vector<BFloat16> layer_output(
      Fixture::kHidden, BFloat16::FromFloat(10.0F));
  std::vector<BFloat16> output(fixture.residual.size());
  ASSERT_TRUE(deepseek_mhc_post_oracle(
      layer_output, fixture.residual, fixture.post, fixture.mix, 1,
      Fixture::kHidden, output).ok());
  for (const auto value : output) {
    EXPECT_FLOAT_EQ(value.to_float(), 12.5F);
  }
}

TEST(DeepSeekMhcOracleTest, SinkhornIterationsApproachDoublyStochasticMix) {
  Fixture fixture;
  for (std::uint32_t index = 0; index < 16; ++index) {
    fixture.base[8 + index] =
        static_cast<float>(static_cast<int>(index) - 8) * 0.125F;
  }
  auto parameters = fixture.parameters();
  parameters.sinkhorn_epsilon = 1.0e-6F;
  parameters.sinkhorn_iterations = 8;
  ASSERT_TRUE(deepseek_mhc_pre_oracle(
      fixture.residual, parameters, 1, Fixture::kHidden, fixture.post,
      fixture.mix, fixture.input).ok());
  for (std::uint32_t row = 0; row < 4; ++row) {
    float sum = 0.0F;
    for (std::uint32_t column = 0; column < 4; ++column) {
      sum += fixture.mix[row * 4 + column];
    }
    EXPECT_NEAR(sum, 1.0F, 2.0e-5F);
  }
  for (std::uint32_t column = 0; column < 4; ++column) {
    float sum = 0.0F;
    for (std::uint32_t row = 0; row < 4; ++row) {
      sum += fixture.mix[row * 4 + column];
    }
    EXPECT_NEAR(sum, 1.0F, 2.0e-5F);
  }
}

TEST(DeepSeekMhcOracleTest, RejectsNonFiniteInputWithoutPublishingOutputs) {
  Fixture fixture;
  fixture.residual[3] = BFloat16::FromFloat(
      std::numeric_limits<float>::quiet_NaN());
  std::fill(fixture.post.begin(), fixture.post.end(), 7.0F);
  std::fill(fixture.mix.begin(), fixture.mix.end(), 7.0F);
  std::fill(fixture.input.begin(), fixture.input.end(),
            BFloat16::FromFloat(7.0F));
  EXPECT_FALSE(deepseek_mhc_pre_oracle(
      fixture.residual, fixture.parameters(), 1, Fixture::kHidden,
      fixture.post, fixture.mix, fixture.input).ok());
  EXPECT_FLOAT_EQ(fixture.post[0], 7.0F);
  EXPECT_FLOAT_EQ(fixture.mix[0], 7.0F);
  EXPECT_FLOAT_EQ(fixture.input[0].to_float(), 7.0F);
}

} }  // namespace pih
