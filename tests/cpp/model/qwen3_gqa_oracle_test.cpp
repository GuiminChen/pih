#include "pih/model/qwen3_gqa_oracle.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

std::vector<BFloat16> bf16(std::size_t count, float value = 0.0F) {
  return std::vector<BFloat16>(count, BFloat16::FromFloat(value));
}

TEST(QwenGqaAttentionOracleTest, AppliesCausalMaskWithoutScoreMatrix) {
  constexpr std::uint32_t query_tokens = 2;
  constexpr std::uint32_t history_tokens = 2;
  auto query = bf16(query_tokens * 16 * 128);
  auto keys = bf16(history_tokens * 8 * 128);
  auto values = bf16(history_tokens * 8 * 128);
  for (std::uint32_t head = 0; head < 8; ++head) {
    for (std::uint32_t column = 0; column < 128; ++column) {
      values[(0 * 8 + head) * 128 + column] =
          BFloat16::FromFloat(static_cast<float>(head + 1));
      values[(1 * 8 + head) * 128 + column] =
          BFloat16::FromFloat(static_cast<float>(head + 3));
    }
  }
  auto output = bf16(query.size(), -99.0F);
  ASSERT_TRUE(qwen_gqa_attention_oracle(query, keys, values, query_tokens,
                                        history_tokens, 0, output)
                  .ok());
  for (std::uint32_t query_head = 0; query_head < 16; ++query_head) {
    const float first = static_cast<float>(query_head / 2 + 1);
    const float second = static_cast<float>(query_head / 2 + 2);
    EXPECT_FLOAT_EQ(output[(0 * 16 + query_head) * 128].to_float(), first);
    EXPECT_FLOAT_EQ(output[(1 * 16 + query_head) * 128].to_float(), second);
  }
}

TEST(QwenGqaAttentionOracleTest, DecodeReadsTheCommittedHistoryPrefix) {
  auto query = bf16(16 * 128);
  auto keys = bf16(3 * 8 * 128);
  auto values = bf16(3 * 8 * 128);
  for (std::uint32_t token = 0; token < 3; ++token) {
    for (std::size_t index = token * 8 * 128;
         index < (token + 1) * 8 * 128; ++index) {
      values[index] = BFloat16::FromFloat(static_cast<float>(token + 1));
    }
  }
  auto output = bf16(query.size());
  ASSERT_TRUE(qwen_gqa_attention_oracle(query, keys, values, 1, 3, 2, output)
                  .ok());
  for (const auto value : output) EXPECT_FLOAT_EQ(value.to_float(), 2.0F);
}

TEST(QwenGqaAttentionOracleTest, RejectsRangeExtentAndNonfiniteAtomically) {
  auto query = bf16(16 * 128);
  auto keys = bf16(8 * 128);
  auto values = bf16(8 * 128, 1.0F);
  auto output = bf16(query.size(), 7.0F);
  EXPECT_FALSE(qwen_gqa_attention_oracle(query, keys, values, 2, 1, 0,
                                          output)
                   .ok());
  query[0] = BFloat16::FromFloat(std::numeric_limits<float>::infinity());
  EXPECT_FALSE(qwen_gqa_attention_oracle(query, keys, values, 1, 1, 0,
                                          output)
                   .ok());
  for (const auto value : output) EXPECT_FLOAT_EQ(value.to_float(), 7.0F);
}

}  // namespace
}  // namespace pih
