#include "pih/model/deepseek_compression_append_plan.h"

#include <gtest/gtest.h>

#include "pih/model/deepseek_attention_pool_geometry.h"

namespace pih { namespace {

const DeepSeekStatePoolGeometry& ratio4_geometry() {
  static const auto rows = DeepSeekStatePoolGeometry::CanonicalBf16();
  return rows[1];
}

const DeepSeekStatePoolGeometry& ratio128_geometry() {
  static const auto rows = DeepSeekStatePoolGeometry::CanonicalBf16();
  return rows[3];
}

TEST(DeepSeekCompressionAppendPlanTest, CrossesRatio4AndPageBoundaries) {
  auto plan = DeepSeekCompressionAppendPlanner::Plan(3, 258,
                                                      ratio4_geometry());
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan->first_new_slot, 0U);
  EXPECT_EQ(plan->new_slot_count, 65U);
  EXPECT_EQ(plan->first_touched_page, 0U);
  EXPECT_EQ(plan->touched_page_count, 2U);
  EXPECT_EQ(plan->remainder_before, 3U);
  EXPECT_EQ(plan->remainder_after, 1U);
}

TEST(DeepSeekCompressionAppendPlanTest, Ratio128KeepsIncompleteRemainder) {
  auto plan = DeepSeekCompressionAppendPlanner::Plan(127, 128,
                                                      ratio128_geometry());
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan->new_slot_count, 1U);
  EXPECT_EQ(plan->first_new_slot, 0U);
  EXPECT_EQ(plan->remainder_before, 127U);
  EXPECT_EQ(plan->remainder_after, 127U);
}

TEST(DeepSeekCompressionAppendPlanTest, IsInvariantToChunkPartition) {
  auto whole = DeepSeekCompressionAppendPlanner::Plan(0, 1025,
                                                       ratio4_geometry());
  auto first = DeepSeekCompressionAppendPlanner::Plan(0, 257,
                                                       ratio4_geometry());
  auto second = DeepSeekCompressionAppendPlanner::Plan(257, 768,
                                                        ratio4_geometry());
  ASSERT_TRUE(whole.ok() && first.ok() && second.ok());
  EXPECT_EQ(first->new_slot_count + second->new_slot_count,
            whole->new_slot_count);
  EXPECT_EQ(second->first_new_slot,
            first->first_new_slot + first->new_slot_count);
  EXPECT_EQ(second->remainder_after, whole->remainder_after);
  EXPECT_EQ(second->source_end, whole->source_end);
}

TEST(DeepSeekCompressionAppendPlanTest, RejectsZeroAndContextOverflow) {
  EXPECT_FALSE(
      DeepSeekCompressionAppendPlanner::Plan(0, 0, ratio4_geometry()).ok());
  EXPECT_FALSE(DeepSeekCompressionAppendPlanner::Plan(
                   1048576, 1, ratio4_geometry())
                   .ok());
}

} }  // namespace pih
