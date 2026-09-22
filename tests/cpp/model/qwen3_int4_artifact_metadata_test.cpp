#include "pih/model/qwen3_int4_artifact_metadata.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest digest_of(std::string_view value) {
  return sha256(std::as_bytes(std::span(value))).value();
}

std::vector<QwenInt4PayloadDigest> payload_digests(
    const QwenInt4ArtifactLayout& layout) {
  std::vector<QwenInt4PayloadDigest> result;
  for (const auto& record : layout.records()) {
    if (record.kind != QwenInt4ArtifactRecordKind::kAlias) {
      result.push_back({record.identity, digest_of(record.identity)});
    }
  }
  return result;
}

TEST(QwenInt4ArtifactMetadataTest, RoundTripsFixedCanonicalRegion) {
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  const QwenInt4ArtifactRoots roots{digest_of("source"), digest_of("binding"),
                                     digest_of("disposition")};
  auto metadata = QwenInt4ArtifactMetadata::Create(
      layout, roots, payload_digests(layout));
  ASSERT_TRUE(metadata.ok()) << metadata.status().message();
  auto region = metadata->serialize_fixed_region();
  ASSERT_TRUE(region.ok()) << region.status().message();
  EXPECT_EQ(region->size(), QwenInt4ArtifactLayout::kMetadataBytes);

  auto parsed = QwenInt4ArtifactMetadata::ParseAndVerify(*region, layout);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(parsed->records().size(), 507U);
  EXPECT_EQ(parsed->roots().source_artifact, roots.source_artifact);
  EXPECT_EQ(parsed->format_id(), "xing-w4a16-sym-g128-v1");
  EXPECT_EQ(parsed->layout_id(), "canonical-nk-low-nibble-k-v1");
  EXPECT_LT(parsed->encoded_bytes(), region->size());
  EXPECT_TRUE(std::all_of(region->begin() + parsed->encoded_bytes(), region->end(),
                          [](std::byte value) { return value == std::byte{0}; }));
}

TEST(QwenInt4ArtifactMetadataTest, RejectsMissingAndDuplicatePayloadDigest) {
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  const QwenInt4ArtifactRoots roots{digest_of("source"), digest_of("binding"),
                                     digest_of("disposition")};
  auto digests = payload_digests(layout);
  digests.pop_back();
  EXPECT_FALSE(QwenInt4ArtifactMetadata::Create(layout, roots, digests).ok());
  digests = payload_digests(layout);
  digests.back() = digests.front();
  EXPECT_FALSE(QwenInt4ArtifactMetadata::Create(layout, roots, digests).ok());
}

TEST(QwenInt4ArtifactMetadataTest, ParserRejectsRecordAndZeroTailMutation) {
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  const QwenInt4ArtifactRoots roots{digest_of("source"), digest_of("binding"),
                                     digest_of("disposition")};
  auto metadata = QwenInt4ArtifactMetadata::Create(
      layout, roots, payload_digests(layout)).value();
  auto record_mutation = metadata.serialize_fixed_region().value();
  record_mutation[160] ^= std::byte{1};
  EXPECT_FALSE(QwenInt4ArtifactMetadata::ParseAndVerify(record_mutation, layout).ok());

  auto tail_mutation = metadata.serialize_fixed_region().value();
  tail_mutation.back() = std::byte{1};
  EXPECT_FALSE(QwenInt4ArtifactMetadata::ParseAndVerify(tail_mutation, layout).ok());
}

}  // namespace
}  // namespace pih
