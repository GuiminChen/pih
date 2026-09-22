#include "pih/model/deepseek_dspark_state_cut.h"

#include <gtest/gtest.h>

#include <array>

namespace pih { namespace {
Sha256Digest root() {
  Sha256Digest value;
  value.bytes[0] = std::byte{1};
  return value;
}
DeepSeekDsparkGreedyVerifyResult verified() {
  constexpr std::array<std::uint32_t, 5> drafts{10, 11, 12, 13, 14};
  constexpr std::array<std::uint32_t, 5> targets{10, 11, 12, 99, 14};
  return deepseek_dspark_greedy_verify(drafts, targets).value();
}
TEST(DeepSeekDsparkStateCutTest,
     NonterminalCutPublishesLastRetainedRecordAsPending) {
  auto cut = compile_deepseek_dspark_state_cut(7, 41, verified(), 3, false,
                                                root());
  ASSERT_TRUE(cut.ok());
  EXPECT_EQ(cut->new_generation, 42U);
  EXPECT_EQ(cut->retained_record_count, 3U);
  EXPECT_EQ(cut->processed_delta, 3U);
  ASSERT_TRUE(cut->pending_input_token.has_value());
  EXPECT_EQ(*cut->pending_input_token, 12U);
  EXPECT_FALSE(cut->terminal_drain);
}
TEST(DeepSeekDsparkStateCutTest,
     TerminalCutKeepsUsageRecordsButDoesNotCreatePendingInput) {
  auto cut = compile_deepseek_dspark_state_cut(7, 41, verified(), 2, true,
                                                root());
  ASSERT_TRUE(cut.ok());
  EXPECT_EQ(cut->retained_record_count, 2U);
  EXPECT_EQ(cut->processed_delta, 2U);
  EXPECT_FALSE(cut->pending_input_token.has_value());
  EXPECT_TRUE(cut->terminal_drain);
}
TEST(DeepSeekDsparkStateCutTest, RejectsForgedCutAndGenerationOverflow) {
  EXPECT_FALSE(compile_deepseek_dspark_state_cut(7, 41, verified(), 5, false,
                                                 root()).ok());
  EXPECT_FALSE(compile_deepseek_dspark_state_cut(
      7, std::numeric_limits<std::uint64_t>::max(), verified(), 1, false,
      root()).ok());
  EXPECT_FALSE(compile_deepseek_dspark_state_cut(7, 41, verified(), 1, false,
                                                 Sha256Digest{}).ok());
}
}}  // namespace pih::<anonymous>
