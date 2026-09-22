#include "pih/model/qwen3_int4_artifact_layout.h"

#include <cstdint>
#include <set>
#include <string>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenInt4ArtifactLayoutTest, FreezesCanonicalPureW4Inventory) {
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4();
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_EQ(layout->records().size(), 507U);
  EXPECT_EQ(layout->payload_record_count(), 506U);
  EXPECT_EQ(layout->logical_payload_bytes(), 538'378'240U);
  EXPECT_EQ(layout->padded_data_bytes(), 538'710'016U);
  EXPECT_EQ(layout->file_bytes(), 539'758'592U);

  std::uint64_t values = 0;
  std::uint64_t scales = 0;
  std::uint64_t embeddings = 0;
  std::uint64_t norms = 0;
  std::uint64_t aliases = 0;
  for (const auto& record : layout->records()) {
    switch (record.kind) {
      case QwenInt4ArtifactRecordKind::kPackedValues: ++values; break;
      case QwenInt4ArtifactRecordKind::kFp16Scales: ++scales; break;
      case QwenInt4ArtifactRecordKind::kBf16Embedding: ++embeddings; break;
      case QwenInt4ArtifactRecordKind::kBf16Norm: ++norms; break;
      case QwenInt4ArtifactRecordKind::kAlias: ++aliases; break;
    }
  }
  EXPECT_EQ(values, 196U);
  EXPECT_EQ(scales, 196U);
  EXPECT_EQ(embeddings, 1U);
  EXPECT_EQ(norms, 113U);
  EXPECT_EQ(aliases, 1U);
}

TEST(QwenInt4ArtifactLayoutTest, AssignsIndependentAlignedZeroPaddedExtents) {
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4();
  ASSERT_TRUE(layout.ok());
  std::uint64_t next_offset = QwenInt4ArtifactLayout::kMetadataBytes;
  std::set<std::string> identities;
  for (const auto& record : layout->records()) {
    EXPECT_TRUE(identities.insert(record.identity).second);
    if (record.kind == QwenInt4ArtifactRecordKind::kAlias) {
      EXPECT_EQ(record.logical_bytes, 0U);
      EXPECT_EQ(record.extent_bytes, 0U);
      EXPECT_EQ(record.alias_owner, "model.embed_tokens.weight");
      continue;
    }
    EXPECT_EQ(record.file_offset % 4096U, 0U);
    EXPECT_EQ(record.extent_bytes % 4096U, 0U);
    EXPECT_GE(record.extent_bytes, record.logical_bytes);
    EXPECT_EQ(record.file_offset, next_offset);
    next_offset += record.extent_bytes;
  }
  EXPECT_EQ(next_offset, layout->file_bytes());
}

TEST(QwenInt4ArtifactLayoutTest, CanonicalOrderIgnoresConstructionOrder) {
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4();
  ASSERT_TRUE(layout.ok());
  ASSERT_FALSE(layout->records().empty());
  EXPECT_EQ(layout->records().front().identity, "lm_head.weight");
  EXPECT_EQ(layout->records().back().identity, "model.norm.weight");
  for (std::size_t index = 1; index < layout->records().size(); ++index) {
    EXPECT_LT(layout->records()[index - 1].identity,
              layout->records()[index].identity);
  }
}

TEST(QwenInt4ArtifactLayoutTest, FreezesStableLayoutIdentity) {
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4();
  ASSERT_TRUE(layout.ok());
  auto digest = layout->semantic_digest();
  ASSERT_TRUE(digest.ok());
  EXPECT_EQ(digest->hex(),
            "984846c28fa5fbd933bd521c8a8a9f996686f91d2a4f1c23c19b3e5d04a277ea");
}

}  // namespace
}  // namespace pih
