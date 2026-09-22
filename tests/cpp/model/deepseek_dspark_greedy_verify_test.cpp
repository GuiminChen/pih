#include "pih/model/deepseek_dspark_greedy_verify.h"

#include <gtest/gtest.h>

#include <array>

namespace pih { namespace {
TEST(DeepSeekDsparkGreedyVerifyTest, CoversEveryAcceptedDraftCountZeroToFive) {
  constexpr std::array<std::uint32_t, 5> drafts{10, 11, 12, 13, 14};
  for (std::uint32_t accepted = 0; accepted <= 5; ++accepted) {
    auto targets = drafts;
    if (accepted < 5) targets[accepted] = 100 + accepted;
    auto result = deepseek_dspark_greedy_verify(drafts, targets);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->accepted_draft_count, accepted);
    EXPECT_EQ(result->retained_record_count,
              accepted == 5 ? 5U : accepted + 1U);
    EXPECT_EQ(result->mismatch, accepted != 5);
    for (std::uint32_t index = 0; index < accepted; ++index)
      EXPECT_EQ(result->retained_tokens[index], drafts[index]);
    if (accepted < 5)
      EXPECT_EQ(result->retained_tokens[accepted], targets[accepted]);
  }
}
TEST(DeepSeekDsparkGreedyVerifyTest, NeverEmitsBonusAfterFiveMatches) {
  constexpr std::array<std::uint32_t, 5> tokens{1, 2, 3, 4, 5};
  auto result = deepseek_dspark_greedy_verify(tokens, tokens);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->retained_record_count, 5U);
  EXPECT_FALSE(result->mismatch);
}
TEST(DeepSeekDsparkGreedyVerifyTest, RejectsUnknownGeometryAndTokenIds) {
  constexpr std::array<std::uint32_t, 1> valid{1};
  constexpr std::array<std::uint32_t, 2> two{1, 2};
  EXPECT_FALSE(deepseek_dspark_greedy_verify({}, {}).ok());
  EXPECT_FALSE(deepseek_dspark_greedy_verify(valid, two).ok());
  constexpr std::array<std::uint32_t, 1> invalid{129280};
  EXPECT_FALSE(deepseek_dspark_greedy_verify(invalid, valid).ok());
  EXPECT_FALSE(deepseek_dspark_greedy_verify(valid, valid, 129279).ok());
}
}}  // namespace pih::<anonymous>
