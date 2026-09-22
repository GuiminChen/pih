#include "pih/model/deepseek_attention_index_assembler.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih { namespace {

TEST(DeepSeekAttentionIndexAssemblerTest,
     PadsRatio4PrefillRowsToOneKernelWidth) {
  const std::vector<std::uint32_t> positions{0, 3, 7};
  const std::vector<std::vector<std::uint32_t>> selected{{}, {0}, {1, 0}};
  auto matrix = DeepSeekAttentionIndexAssembler::Ratio4(
      positions, selected, 2, 10, 1000);
  ASSERT_TRUE(matrix.ok());
  EXPECT_EQ(matrix->query_count, 3U);
  EXPECT_EQ(matrix->row_width, 130U);
  EXPECT_EQ(matrix->values[0], 10);
  EXPECT_EQ(matrix->values[1], -1);
  EXPECT_EQ(matrix->values[128], -1);
  EXPECT_EQ(matrix->values[130 + 128], 1000);
  EXPECT_EQ(matrix->values[260 + 128], 1001);
  EXPECT_EQ(matrix->values[260 + 129], 1000);
}

TEST(DeepSeekAttentionIndexAssemblerTest,
     PadsRatio128CausalPrefixAndPreservesRecentWrap) {
  const std::vector<std::uint32_t> positions{126, 127, 255};
  auto matrix = DeepSeekAttentionIndexAssembler::Ratio128(
      positions, 2, 0, 128);
  ASSERT_TRUE(matrix.ok());
  EXPECT_EQ(matrix->row_width, 130U);
  EXPECT_EQ(matrix->values[128], -1);
  EXPECT_EQ(matrix->values[130 + 128], 128);
  EXPECT_EQ(matrix->values[260], 0);
  EXPECT_EQ(matrix->values[260 + 127], 127);
  EXPECT_EQ(matrix->values[260 + 128], 128);
  EXPECT_EQ(matrix->values[260 + 129], 129);
}

TEST(DeepSeekAttentionIndexAssemblerTest, RejectsDuplicateAndFutureSelection) {
  const std::vector<std::uint32_t> positions{7};
  const std::vector<std::vector<std::uint32_t>> duplicate{{0, 0}};
  EXPECT_FALSE(DeepSeekAttentionIndexAssembler::Ratio4(
                   positions, duplicate, 2, 0, 128)
                   .ok());
  const std::vector<std::vector<std::uint32_t>> future{{2}};
  EXPECT_FALSE(DeepSeekAttentionIndexAssembler::Ratio4(
                   positions, future, 2, 0, 128)
                   .ok());
}

TEST(DeepSeekAttentionIndexAssemblerTest, EnforcesMaximumRatio128Width) {
  const std::vector<std::uint32_t> positions{1048575};
  auto matrix = DeepSeekAttentionIndexAssembler::Ratio128(
      positions, 8192, 0, 128);
  ASSERT_TRUE(matrix.ok());
  EXPECT_EQ(matrix->row_width, 8320U);
  EXPECT_FALSE(DeepSeekAttentionIndexAssembler::Ratio128(
                   positions, 8193, 0, 128)
                   .ok());
}

TEST(DeepSeekAttentionIndexAssemblerTest, RejectsOverlappingPhysicalKvRanges) {
  const std::vector<std::uint32_t> positions{127};
  const std::vector<std::vector<std::uint32_t>> selected{{0}};
  EXPECT_FALSE(DeepSeekAttentionIndexAssembler::Ratio4(
                   positions, selected, 1, 100, 200)
                   .ok());
  EXPECT_TRUE(DeepSeekAttentionIndexAssembler::Ratio4(
                  positions, selected, 1, 100, 228)
                  .ok());
}

} }  // namespace pih
