#include "pih/model/deepseek_sparse_attention_oracle.h"

#include <gtest/gtest.h>

#include <cmath>

namespace pih { namespace {

TEST(DeepSeekSparseAttentionOracleTest, IncludesSinkOnlyInDenominator) {
  std::vector<BFloat16> query(512, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> kv(2U * 512U);
  std::fill_n(kv.begin(), 512, BFloat16::FromFloat(1.0F));
  std::fill_n(kv.begin() + 512, 512, BFloat16::FromFloat(3.0F));
  const float sink[] = {0.0F};
  const std::int32_t indices[] = {0, 1};
  auto output = DeepSeekSparseAttentionOracle::Evaluate(
      query, kv, sink, indices, 1, 2, 1.0F / std::sqrt(512.0F));
  ASSERT_TRUE(output.ok());
  EXPECT_FLOAT_EQ((*output)[0].to_float(),
                  BFloat16::FromFloat(4.0F / 3.0F).to_float());
}

TEST(DeepSeekSparseAttentionOracleTest, IgnoresMinusOnePadding) {
  std::vector<BFloat16> query(512, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> kv(512, BFloat16::FromFloat(2.0F));
  const float sink[] = {0.0F};
  const std::int32_t indices[] = {-1, 0, -1};
  auto output = DeepSeekSparseAttentionOracle::Evaluate(
      query, kv, sink, indices, 1, 1, 1.0F / std::sqrt(512.0F));
  ASSERT_TRUE(output.ok());
  EXPECT_FLOAT_EQ((*output)[511].to_float(), 1.0F);
}

TEST(DeepSeekSparseAttentionOracleTest, PreservesDuplicateIndexSemantics) {
  std::vector<BFloat16> query(512, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> kv(512, BFloat16::FromFloat(3.0F));
  const float sink[] = {0.0F};
  const std::int32_t indices[] = {0, 0};
  auto output = DeepSeekSparseAttentionOracle::Evaluate(
      query, kv, sink, indices, 1, 1, 1.0F / std::sqrt(512.0F));
  ASSERT_TRUE(output.ok());
  EXPECT_FLOAT_EQ((*output)[0].to_float(), 2.0F);
}

TEST(DeepSeekSparseAttentionOracleTest, RejectsInvalidVisibility) {
  std::vector<BFloat16> query(512, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> kv(512, BFloat16::FromFloat(0.0F));
  const float sink[] = {0.0F};
  const std::int32_t hidden[] = {-1};
  EXPECT_FALSE(DeepSeekSparseAttentionOracle::Evaluate(
                   query, kv, sink, hidden, 1, 1,
                   1.0F / std::sqrt(512.0F))
                   .ok());
  const std::int32_t out_of_range[] = {1};
  EXPECT_FALSE(DeepSeekSparseAttentionOracle::Evaluate(
                   query, kv, sink, out_of_range, 1, 1,
                   1.0F / std::sqrt(512.0F))
                   .ok());
}

TEST(DeepSeekSparseAttentionOracleTest, SupportsRatio128MaximumIndexEnvelope) {
  std::vector<BFloat16> query(512, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> kv(512, BFloat16::FromFloat(1.0F));
  std::vector<std::int32_t> indices(8320, 0);
  const float sink[] = {0.0F};
  auto output = DeepSeekSparseAttentionOracle::Evaluate(
      query, kv, sink, indices, 1, 1, 1.0F / std::sqrt(512.0F));
  ASSERT_TRUE(output.ok());
  EXPECT_FLOAT_EQ(output->front().to_float(),
                  BFloat16::FromFloat(8320.0F / 8321.0F).to_float());
  indices.push_back(0);
  EXPECT_FALSE(DeepSeekSparseAttentionOracle::Evaluate(
                   query, kv, sink, indices, 1, 1,
                   1.0F / std::sqrt(512.0F))
                   .ok());
}

TEST(DeepSeekSparseAttentionOracleTest,
     PagedViewMatchesLogicalViewAcrossNoncontiguousPages) {
  std::vector<BFloat16> query(512, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> recent(128U * 512U, BFloat16::FromFloat(1.0F));
  std::vector<BFloat16> compressed(2U * 64U * 512U);
  std::fill_n(compressed.begin(), 64U * 512U,
              BFloat16::FromFloat(7.0F));
  std::fill_n(compressed.begin() + 64U * 512U, 64U * 512U,
              BFloat16::FromFloat(3.0F));
  std::vector<BFloat16> logical(193U * 512U);
  std::copy(recent.begin(), recent.end(), logical.begin());
  std::fill_n(logical.begin() + 128U * 512U, 64U * 512U,
              BFloat16::FromFloat(3.0F));
  std::fill_n(logical.begin() + 192U * 512U, 512U,
              BFloat16::FromFloat(7.0F));
  const std::uint32_t pages[] = {1, 0};
  const float sink[] = {0.0F};
  const std::int32_t indices[] = {0, 128, 192};
  const auto scale = 1.0F / std::sqrt(512.0F);
  auto expected = DeepSeekSparseAttentionOracle::Evaluate(
      query, logical, sink, indices, 1, 193, scale);
  auto actual = DeepSeekSparseAttentionOracle::EvaluatePaged(
      query, recent, compressed, pages, sink, indices, 1, 193,
      0, 128, 65, 2, scale);
  ASSERT_TRUE(expected.ok() && actual.ok());
  EXPECT_EQ(*actual, *expected);
  const std::uint32_t bad_pages[] = {1, 2};
  EXPECT_FALSE(DeepSeekSparseAttentionOracle::EvaluatePaged(
                   query, recent, compressed, bad_pages, sink, indices,
                   1, 193, 0, 128, 65, 2, scale).ok());
}

} }  // namespace pih
