#include "pih/model/deepseek_dspark_prefill_reference.h"

#include <gtest/gtest.h>

#include <array>

namespace pih {
namespace {

TEST(DeepSeekDsparkPrefillReferenceTest,
     ChunkingDoesNotChangeAnyStageRecentRing) {
  const auto whole = build_deepseek_dspark_prefill_reference().value();
  const std::array<std::uint32_t, 3> chunks{2, 3, 2};
  const auto chunked =
      build_deepseek_dspark_prefill_reference(chunks).value();
  EXPECT_EQ(whole.physical_recent_state, chunked.physical_recent_state);
  EXPECT_EQ(whole.logical_recent_state, chunked.logical_recent_state);
  EXPECT_EQ(whole.logical_positions,
            (std::array<std::uint32_t, 4>{3, 4, 5, 6}));
  EXPECT_EQ(whole.write_counts,
            (std::array<std::uint32_t, 3>{7, 7, 7}));
}

TEST(DeepSeekDsparkPrefillReferenceTest,
     FreezesStageSpecificPhysicalAndLogicalValues) {
  const auto receipt = build_deepseek_dspark_prefill_reference().value();
  constexpr std::array<std::uint16_t, 3> physical_first{
      16087, 16025, 16076};
  constexpr std::array<std::uint16_t, 3> physical_last{
      16357, 16375, 16384};
  constexpr std::array<std::uint16_t, 3> logical_first{
      16086, 16025, 16076};
  constexpr std::array<std::uint16_t, 3> logical_last{
      16369, 16376, 16388};
  for (std::uint32_t stage = 0; stage < kDeepSeekDsparkStageCount; ++stage) {
    ASSERT_EQ(receipt.physical_recent_state[stage].size(), 16U);
    ASSERT_EQ(receipt.logical_recent_state[stage].size(), 16U);
    EXPECT_EQ(receipt.physical_recent_state[stage].front().bits,
              physical_first[stage]);
    EXPECT_EQ(receipt.physical_recent_state[stage].back().bits,
              physical_last[stage]);
    EXPECT_EQ(receipt.logical_recent_state[stage].front().bits,
              logical_first[stage]);
    EXPECT_EQ(receipt.logical_recent_state[stage].back().bits,
              logical_last[stage]);
  }
  EXPECT_NE(receipt.physical_recent_state[0],
            receipt.physical_recent_state[1]);
  EXPECT_NE(receipt.physical_recent_state[1],
            receipt.physical_recent_state[2]);
}

TEST(DeepSeekDsparkPrefillReferenceTest,
     RejectsZeroOversizedAndIncompleteChunkSchedules) {
  const std::array<std::uint32_t, 2> zero{0, 7};
  const std::array<std::uint32_t, 1> oversized{8};
  const std::array<std::uint32_t, 2> incomplete{2, 4};
  EXPECT_FALSE(build_deepseek_dspark_prefill_reference(zero).ok());
  EXPECT_FALSE(build_deepseek_dspark_prefill_reference(oversized).ok());
  EXPECT_FALSE(build_deepseek_dspark_prefill_reference(incomplete).ok());
}

}  // namespace
}  // namespace pih
