#include "pih/model/deepseek_attention_pool_geometry.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekAttentionPoolGeometryTest, FreezesCanonicalBf16Rows) {
  const auto rows = DeepSeekStatePoolGeometry::CanonicalBf16();
  ASSERT_EQ(rows.size(), 4U);
  for (const auto& row : rows) EXPECT_TRUE(row.validate().ok());
  EXPECT_EQ(rows[0].physical_page_bytes, 131072U);
  EXPECT_EQ(rows[1].logical_coverage_tokens, 256U);
  EXPECT_EQ(rows[1].handles_per_logical_page, 2U);
  EXPECT_EQ(rows[2].physical_page_bytes, 16384U);
  EXPECT_EQ(rows[3].logical_coverage_tokens, 8192U);
}

TEST(DeepSeekAttentionPoolGeometryTest, RoundTripsExactPaddingFreeWire) {
  for (const auto& row : DeepSeekStatePoolGeometry::CanonicalBf16()) {
    auto wire = row.encode();
    ASSERT_TRUE(wire.ok());
    EXPECT_EQ(wire->size(), 48U);
    auto decoded = DeepSeekStatePoolGeometry::Decode(*wire);
    ASSERT_TRUE(decoded.ok());
    EXPECT_EQ(decoded->physical_page_bytes, row.physical_page_bytes);
    EXPECT_EQ(decoded->paired_commit_group, row.paired_commit_group);
  }
}

TEST(DeepSeekAttentionPoolGeometryTest, RejectsAmbiguousOrReferenceGeometry) {
  auto row = DeepSeekStatePoolGeometry::CanonicalBf16()[1];
  row.logical_coverage_tokens = 64;
  EXPECT_FALSE(row.validate().ok());
  row = DeepSeekStatePoolGeometry::CanonicalBf16()[1];
  row.bytes_per_storage_unit = 584;
  row.raw_page_bytes = 37376;
  row.page_alignment_bytes = 512;
  row.physical_page_bytes = 37376;
  EXPECT_FALSE(row.validate().ok());
  row = DeepSeekStatePoolGeometry::CanonicalBf16()[2];
  row.paired_commit_group = 0;
  EXPECT_FALSE(row.validate().ok());
}

} }  // namespace pih
