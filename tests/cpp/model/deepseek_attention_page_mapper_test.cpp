#include "pih/model/deepseek_attention_page_mapper.h"

#include <gtest/gtest.h>

namespace pih { namespace {

const DeepSeekStatePoolGeometry& ratio4() {
  static const auto rows = DeepSeekStatePoolGeometry::CanonicalBf16();
  return rows[1];
}

const DeepSeekStatePoolGeometry& ratio128() {
  static const auto rows = DeepSeekStatePoolGeometry::CanonicalBf16();
  return rows[3];
}

TEST(DeepSeekAttentionPageMapperTest, PreservesRatio4FloorAndPageBoundaries) {
  auto before = DeepSeekAttentionPageMapper::Coverage(255, ratio4());
  auto boundary = DeepSeekAttentionPageMapper::Coverage(256, ratio4());
  auto after = DeepSeekAttentionPageMapper::Coverage(257, ratio4());
  ASSERT_TRUE(before.ok() && boundary.ok() && after.ok());
  EXPECT_EQ(before->stored_slots, 63U);
  EXPECT_EQ(before->physical_pages, 1U);
  EXPECT_EQ(before->compressor_remainder_tokens, 3U);
  EXPECT_EQ(boundary->stored_slots, 64U);
  EXPECT_EQ(boundary->physical_pages, 1U);
  EXPECT_EQ(after->stored_slots, 64U);
  EXPECT_EQ(after->compressor_remainder_tokens, 1U);
  EXPECT_EQ(DeepSeekAttentionPageMapper::Coverage(260, ratio4())
                ->physical_pages,
            2U);
}

TEST(DeepSeekAttentionPageMapperTest, PreservesRatio128Boundaries) {
  auto before = DeepSeekAttentionPageMapper::Coverage(8191, ratio128());
  auto boundary = DeepSeekAttentionPageMapper::Coverage(8192, ratio128());
  auto after = DeepSeekAttentionPageMapper::Coverage(8193, ratio128());
  ASSERT_TRUE(before.ok() && boundary.ok() && after.ok());
  EXPECT_EQ(before->stored_slots, 63U);
  EXPECT_EQ(before->compressor_remainder_tokens, 127U);
  EXPECT_EQ(boundary->stored_slots, 64U);
  EXPECT_EQ(boundary->physical_pages, 1U);
  EXPECT_EQ(after->stored_slots, 64U);
  EXPECT_EQ(after->compressor_remainder_tokens, 1U);
}

TEST(DeepSeekAttentionPageMapperTest, MapsStoredSlotsNotSourceTokens) {
  auto last = DeepSeekAttentionPageMapper::LocateStoredSlot(63, ratio4());
  auto next = DeepSeekAttentionPageMapper::LocateStoredSlot(64, ratio4());
  ASSERT_TRUE(last.ok() && next.ok());
  EXPECT_EQ(last->page_ordinal, 0U);
  EXPECT_EQ(last->slot_in_page, 63U);
  EXPECT_EQ(next->page_ordinal, 1U);
  EXPECT_EQ(next->slot_in_page, 0U);
  const auto recent = DeepSeekStatePoolGeometry::CanonicalBf16()[0];
  EXPECT_FALSE(DeepSeekAttentionPageMapper::Coverage(128, recent).ok());
}

} }  // namespace pih
