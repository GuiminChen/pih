#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_cpu_oracle.h"

namespace pih {
namespace {

BFloat16 bf(float value) { return BFloat16::FromFloat(value); }

TEST(Qwen3CpuOracleTest, EmbeddingGathersRowsAndRejectsIdsAtomically) {
  const std::array<BFloat16, 6> table{bf(1), bf(2), bf(3), bf(4), bf(5), bf(6)};
  std::array<BFloat16, 4> output{};
  const std::array<std::int64_t, 2> ids{2, 0};
  ASSERT_TRUE(qwen_embedding_oracle(table, 3, 2, ids, output).ok());
  EXPECT_EQ(output, (std::array<BFloat16, 4>{bf(5), bf(6), bf(1), bf(2)}));
  output.fill(bf(9));
  const std::array<std::int64_t, 2> bad_ids{1, 3};
  EXPECT_FALSE(qwen_embedding_oracle(table, 3, 2, bad_ids, output).ok());
  EXPECT_EQ(output[0], bf(9));
}

TEST(Qwen3CpuOracleTest, RmsNormUsesFp32ReductionAndBf16Writeback) {
  const std::array<BFloat16, 4> input{bf(3), bf(4), bf(0), bf(0)};
  const std::array<BFloat16, 2> weight{bf(1), bf(2)};
  std::array<BFloat16, 4> output{};
  ASSERT_TRUE(qwen_rms_norm_oracle(input, weight, 2, 2, 1.0F, output).ok());
  EXPECT_EQ(output[0], bf(3.0F / std::sqrt(13.5F)));
  EXPECT_EQ(output[1], bf(8.0F / std::sqrt(13.5F)));
  EXPECT_EQ(output[2], bf(0));
  EXPECT_EQ(output[3], bf(0));
  EXPECT_FALSE(qwen_rms_norm_oracle(input, weight, 2, 2, 0.0F, output).ok());
}

TEST(Qwen3CpuOracleTest, ResidualAndSiluRoundOnlyAtOutput) {
  const std::array<BFloat16, 3> lhs{bf(1), bf(-2), bf(0.5F)};
  const std::array<BFloat16, 3> rhs{bf(2), bf(3), bf(-4)};
  std::array<BFloat16, 3> output{};
  ASSERT_TRUE(qwen_residual_add_oracle(lhs, rhs, output).ok());
  EXPECT_EQ(output, (std::array<BFloat16, 3>{bf(3), bf(1), bf(-3.5F)}));
  ASSERT_TRUE(qwen_silu_mul_oracle(lhs, rhs, output).ok());
  for (std::size_t index = 0; index < output.size(); ++index) {
    const float gate = lhs[index].to_float();
    EXPECT_EQ(output[index],
              bf((gate / (1.0F + std::exp(-gate))) * rhs[index].to_float()));
  }
}

TEST(Qwen3CpuOracleTest, RopeUsesQwenRotateHalfConventionAndSupportsAliasing) {
  std::array<BFloat16, 4> values{bf(1), bf(2), bf(3), bf(4)};
  const std::array<float, 2> cosine{0.0F, 1.0F};
  const std::array<float, 2> sine{1.0F, 0.0F};
  ASSERT_TRUE(qwen_rope_oracle(values, cosine, sine, 1, 1, 4, values).ok());
  EXPECT_EQ(values, (std::array<BFloat16, 4>{bf(-3), bf(2), bf(1), bf(4)}));
  EXPECT_FALSE(qwen_rope_oracle(values, cosine, sine, 1, 1, 3, values).ok());
}

TEST(Qwen3CpuOracleTest, RopeAnglesAreSharedAcrossHeadsPerToken) {
  const std::array<std::int64_t, 2> positions{0, 1};
  std::array<float, 128> cosine{};
  std::array<float, 128> sine{};
  ASSERT_TRUE(qwen_rope_angles_oracle(positions, 128, 1'000'000.0, cosine,
                                      sine)
                  .ok());
  for (std::size_t pair = 0; pair < 64; ++pair) {
    EXPECT_FLOAT_EQ(cosine[pair], 1.0F);
    EXPECT_FLOAT_EQ(sine[pair], 0.0F);
  }
  EXPECT_FLOAT_EQ(cosine[64], std::cos(1.0F));
  EXPECT_FLOAT_EQ(sine[64], std::sin(1.0F));
  EXPECT_EQ(cosine.size() * sizeof(float) + sine.size() * sizeof(float),
            positions.size() * 512);
}

TEST(Qwen3CpuOracleTest, RopeBroadcastsOneAngleRowAcrossAllHeads) {
  std::array<BFloat16, 8> values{bf(1), bf(2), bf(3), bf(4),
                                  bf(5), bf(6), bf(7), bf(8)};
  const std::array<float, 2> cosine{0.0F, 1.0F};
  const std::array<float, 2> sine{1.0F, 0.0F};
  ASSERT_TRUE(qwen_rope_oracle(values, cosine, sine, 1, 2, 4, values).ok());
  EXPECT_FLOAT_EQ(values[0].to_float(), -3.0F);
  EXPECT_FLOAT_EQ(values[2].to_float(), 1.0F);
  EXPECT_FLOAT_EQ(values[4].to_float(), -7.0F);
  EXPECT_FLOAT_EQ(values[6].to_float(), 5.0F);
}

TEST(Qwen3CpuOracleTest, RopeAnglesRejectInvalidGeometryAtomically) {
  const std::array<std::int64_t, 1> invalid_position{40960};
  std::array<float, 64> cosine;
  std::array<float, 64> sine;
  cosine.fill(7.0F);
  sine.fill(9.0F);
  EXPECT_FALSE(qwen_rope_angles_oracle(invalid_position, 128, 1'000'000.0,
                                       cosine, sine)
                   .ok());
  EXPECT_FLOAT_EQ(cosine.front(), 7.0F);
  EXPECT_FLOAT_EQ(sine.back(), 9.0F);
  EXPECT_FALSE(qwen_rope_angles_oracle({invalid_position.data(), 0}, 128,
                                       1'000'000.0, {}, {})
                   .ok());
}

TEST(Qwen3CpuOracleTest, GreedyArgmaxUsesLowestTokenTieBreak) {
  std::vector<float> logits(151936, -4.0F);
  logits[7] = 3.0F;
  logits[19] = 3.0F;
  auto token = qwen_greedy_argmax_oracle(logits);
  ASSERT_TRUE(token.ok());
  EXPECT_EQ(*token, 7);
}

TEST(Qwen3CpuOracleTest, GreedyArgmaxRejectsExtentAndNonfinite) {
  EXPECT_FALSE(qwen_greedy_argmax_oracle({}).ok());
  std::vector<float> logits(151936, 0.0F);
  logits[123] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(qwen_greedy_argmax_oracle(logits).ok());
}

TEST(Qwen3CpuOracleTest, NonfiniteResultFailsWithoutPublishingPartialOutput) {
  const std::array<BFloat16, 1> lhs{bf(std::numeric_limits<float>::infinity())};
  const std::array<BFloat16, 1> rhs{bf(1)};
  std::array<BFloat16, 1> output{bf(7)};
  EXPECT_FALSE(qwen_residual_add_oracle(lhs, rhs, output).ok());
  EXPECT_EQ(output[0], bf(7));
}

}  // namespace
}  // namespace pih
