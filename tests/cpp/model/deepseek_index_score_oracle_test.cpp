#include "pih/model/deepseek_index_score_oracle.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih { namespace {

TEST(DeepSeekIndexScoreOracleTest, AppliesReluBeforeWeightedHeadReduction) {
  std::vector<BFloat16> query(2U * 128U, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> kv(2U * 128U, BFloat16::FromFloat(0.0F));
  query[0] = BFloat16::FromFloat(2.0F);
  query[128] = BFloat16::FromFloat(-3.0F);
  kv[0] = BFloat16::FromFloat(4.0F);
  kv[128] = BFloat16::FromFloat(5.0F);
  const std::vector<float> weights{0.5F, 7.0F};
  auto score = DeepSeekIndexScoreOracle::Evaluate(query, kv, weights, 1, 2, 2);
  ASSERT_TRUE(score.ok());
  ASSERT_EQ(score->size(), 2U);
  EXPECT_FLOAT_EQ((*score)[0], 4.0F);
  EXPECT_FLOAT_EQ((*score)[1], 5.0F);
}

TEST(DeepSeekIndexScoreOracleTest, KeepsSignedHeadWeightsAfterRelu) {
  std::vector<BFloat16> query(128, BFloat16::FromFloat(1.0F));
  std::vector<BFloat16> kv(128, BFloat16::FromFloat(1.0F));
  const std::vector<float> weights{-0.25F};
  auto score = DeepSeekIndexScoreOracle::Evaluate(query, kv, weights, 1, 1, 1);
  ASSERT_TRUE(score.ok());
  EXPECT_FLOAT_EQ(score->front(), -32.0F);
}

TEST(DeepSeekIndexScoreOracleTest, RejectsMismatchedShape) {
  const std::vector<BFloat16> empty;
  const std::vector<float> empty_weights;
  EXPECT_FALSE(
      DeepSeekIndexScoreOracle::Evaluate(
          empty, empty, empty_weights, 1, 1, 1).ok());
}

TEST(DeepSeekIndexScoreOracleTest, ResolvesNonContiguousPhysicalPages) {
  std::vector<BFloat16> query(128, BFloat16::FromFloat(1.0F));
  std::vector<BFloat16> physical(3U * 64U * 128U,
                                 BFloat16::FromFloat(0.0F));
  for (std::uint32_t column = 0; column < 128; ++column) {
    physical[(2U * 64U + 63U) * 128U + column] =
        BFloat16::FromFloat(2.0F);
    physical[column] = BFloat16::FromFloat(3.0F);
  }
  const std::vector<float> weights{1.0F};
  const std::vector<std::uint32_t> pages{2, 0};
  auto scores = DeepSeekIndexScoreOracle::EvaluatePaged(
      query, physical, weights, pages, 1, 1, 63, 2, 3);
  ASSERT_TRUE(scores.ok()) << scores.status().message();
  ASSERT_EQ(scores->size(), 2U);
  EXPECT_FLOAT_EQ((*scores)[0], 256.0F);
  EXPECT_FLOAT_EQ((*scores)[1], 384.0F);
  const std::vector<std::uint32_t> invalid_pages{3};
  EXPECT_FALSE(DeepSeekIndexScoreOracle::EvaluatePaged(
      query, physical, weights, invalid_pages, 1, 1, 0, 1, 3).ok());
}

} }  // namespace pih
