#include "pih/model/qwen3_int4_artifact_io.h"
#include "../../../plugins/offline-qwen/qwen3_int4_artifact_writer.h"

#include <span>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest io_digest(std::string_view value) {
  return sha256(std::as_bytes(std::span(value))).value();
}

QwenInt4ArtifactMetadata metadata_for(const QwenInt4ArtifactLayout& layout) {
  std::vector<QwenInt4PayloadDigest> digests;
  for (const auto& record : layout.records()) {
    if (record.kind != QwenInt4ArtifactRecordKind::kAlias) {
      digests.push_back({record.identity, io_digest(record.identity)});
    }
  }
  return QwenInt4ArtifactMetadata::Create(
      layout, {io_digest("source"), io_digest("binding"),
               io_digest("disposition")}, std::move(digests)).value();
}

TEST(QwenInt4ArtifactIoTest, ProjectsExactly506CanonicalPayloadExtents) {
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto write_plan = qwen_int4_write_plan(layout);
  auto verify_plan = qwen_int4_verification_plan(metadata_for(layout), layout);
  ASSERT_TRUE(write_plan.ok());
  ASSERT_TRUE(verify_plan.ok());
  ASSERT_EQ(write_plan->size(), 506U);
  ASSERT_EQ(verify_plan->size(), 506U);
  EXPECT_EQ(write_plan->front().file_offset,
            QwenInt4ArtifactLayout::kMetadataBytes);
  const auto& last = write_plan->back();
  EXPECT_EQ(last.file_offset + last.extent_bytes,
            QwenInt4ArtifactLayout::kOfficialFileBytes);
  for (const auto& record : *write_plan) EXPECT_NE(record.identity, "lm_head.weight");
}

TEST(QwenInt4ArtifactIoTest, WholeFileVerifierRejectsNoncanonicalLengthFirst) {
  const std::vector<std::byte> short_file(
      QwenInt4ArtifactLayout::kMetadataBytes, std::byte{0});
  auto verified = verify_qwen_int4_artifact(short_file);
  ASSERT_FALSE(verified.ok());
  EXPECT_EQ(verified.status().code(), StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
