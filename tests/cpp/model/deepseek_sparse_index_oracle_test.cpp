#include "pih/model/deepseek_sparse_index_oracle.h"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace pih { namespace {

TEST(DeepSeekSparseIndexOracleTest, PadsPartialRecentWindow) {
  auto indices = DeepSeekSparseIndexOracle::RecentWindow(2, 100);
  ASSERT_TRUE(indices.ok());
  ASSERT_EQ(indices->size(), 128U);
  EXPECT_EQ((*indices)[0], 100);
  EXPECT_EQ((*indices)[1], 101);
  EXPECT_EQ((*indices)[2], 102);
  EXPECT_EQ((*indices)[3], -1);
}

TEST(DeepSeekSparseIndexOracleTest, WrapsFullRecentWindowChronologically) {
  auto indices = DeepSeekSparseIndexOracle::RecentWindow(128);
  ASSERT_TRUE(indices.ok());
  EXPECT_EQ(indices->front(), 1);
  EXPECT_EQ((*indices)[126], 127);
  EXPECT_EQ(indices->back(), 0);
}

TEST(DeepSeekSparseIndexOracleTest, PublishesOnlyCompleteRatio128Slots) {
  EXPECT_TRUE(DeepSeekSparseIndexOracle::Ratio128(126)->empty());
  auto first = DeepSeekSparseIndexOracle::Ratio128(127, 500);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(*first, std::vector<std::int32_t>({500}));
  auto maximum = DeepSeekSparseIndexOracle::Ratio128(1048575);
  ASSERT_TRUE(maximum.ok());
  EXPECT_EQ(maximum->size(), 8192U);
  EXPECT_FALSE(DeepSeekSparseIndexOracle::Ratio128(1048576).ok());
}

TEST(DeepSeekSparseIndexOracleTest, SelectsVisibleRatio4ScoresInOrder) {
  const std::vector<float> scores{0.5F, 9.0F, 3.0F, 100.0F};
  auto indices = DeepSeekSparseIndexOracle::Ratio4TopK(scores, 3, 20);
  ASSERT_TRUE(indices.ok());
  EXPECT_EQ(*indices, std::vector<std::int32_t>({21, 22, 20}));
}

TEST(DeepSeekSparseIndexOracleTest, RejectsNonfiniteAndBoundaryTies) {
  std::vector<float> scores(513);
  for (std::uint32_t index = 0; index < scores.size(); ++index) {
    scores[index] = static_cast<float>(scores.size() - index);
  }
  scores[511] = scores[512];
  EXPECT_FALSE(
      DeepSeekSparseIndexOracle::Ratio4TopK(scores, scores.size()).ok());
  scores[512] = 0.0F;
  scores[0] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(
      DeepSeekSparseIndexOracle::Ratio4TopK(scores, scores.size()).ok());
}

} }  // namespace pih
