#include "pih/model/deepseek_online_topk.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih { namespace {

TEST(DeepSeekOnlineTopKTest, IsInvariantToTilePartition) {
  std::vector<float> scores(900);
  for (std::uint32_t i = 0; i < scores.size(); ++i) {
    scores[i] = static_cast<float>((i * 37U) % 1009U);
  }
  DeepSeekOnlineTopK single;
  DeepSeekOnlineTopK tiled;
  ASSERT_TRUE(single.consume(0, scores).ok());
  ASSERT_TRUE(tiled.consume(0, std::span<const float>(scores).first(127)).ok());
  ASSERT_TRUE(tiled.consume(127, std::span<const float>(scores).subspan(127, 511)).ok());
  ASSERT_TRUE(tiled.consume(638, std::span<const float>(scores).subspan(638)).ok());
  ASSERT_TRUE(single.finish().ok());
  ASSERT_TRUE(tiled.finish().ok());
  EXPECT_EQ(*single.finish(), *tiled.finish());
  EXPECT_EQ(single.finish()->size(), 512U);
}

TEST(DeepSeekOnlineTopKTest, RejectsGapNonfiniteAndBoundaryTie) {
  DeepSeekOnlineTopK gap;
  const std::vector<float> one{1.0F};
  EXPECT_FALSE(gap.consume(1, one).ok());

  std::vector<float> scores(513);
  for (std::uint32_t i = 0; i < scores.size(); ++i) {
    scores[i] = static_cast<float>(scores.size() - i);
  }
  scores[512] = scores[511];
  DeepSeekOnlineTopK tied;
  ASSERT_TRUE(tied.consume(0, scores).ok());
  EXPECT_FALSE(tied.finish().ok());
}

} }  // namespace pih
