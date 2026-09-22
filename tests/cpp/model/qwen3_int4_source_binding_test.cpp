#include "pih/model/qwen3_int4_source_binding.h"

#include <cstddef>
#include <algorithm>
#include <span>
#include <string>

#include <gtest/gtest.h>

#include "pih/model/qwen3_manifest.h"

namespace pih {
namespace {

struct Fixture final {
  QwenInt4DispositionPlan plan;
  std::vector<QwenInt4ObservedSourceRecord> observed;
  Qwen3SourceArtifactReceipt receipt;
};

Fixture make_fixture() {
  auto plan = QwenInt4DispositionPlan::CreateOfficialPureW4().value();
  std::vector<QwenInt4ObservedSourceRecord> observed;
  std::uint64_t offset = 4096;
  for (const auto& record : plan.records()) {
    const auto digest = sha256(std::as_bytes(std::span(record.source_name))).value();
    observed.push_back({record.source_name, DType::kBFloat16,
                        record.source_shape, offset,
                        offset + record.source_bytes, digest});
    offset += record.source_bytes;
  }
  auto& embedding = *std::find_if(observed.begin(), observed.end(), [](const auto& r) {
    return r.name == "model.embed_tokens.weight";
  });
  auto& head = *std::find_if(observed.begin(), observed.end(), [](const auto& r) {
    return r.name == "lm_head.weight";
  });
  head.payload_sha256 = embedding.payload_sha256;
  Qwen3SourceArtifactReceipt receipt{
      offset, observed.size(), Qwen3Manifest::kOfficialSourcePayloadBytes, {},
      embedding.file_end - embedding.file_begin,
      embedding.file_begin, embedding.file_end, embedding.payload_sha256,
      head.file_begin, head.file_end, head.payload_sha256};
  return {std::move(plan), std::move(observed), receipt};
}

TEST(QwenInt4SourceBindingTest, SealsAllOfficialRangesAndDigests) {
  auto fixture = make_fixture();
  auto binding = QwenInt4SourceBinding::Verify(
      fixture.plan, fixture.observed, fixture.receipt);
  ASSERT_TRUE(binding.ok()) << binding.status().message();
  EXPECT_EQ(binding->records().size(), 311U);
  auto digest = binding->semantic_digest();
  ASSERT_TRUE(digest.ok());
  EXPECT_NE(*digest, Sha256Digest{});
}

TEST(QwenInt4SourceBindingTest, RejectsMissingDuplicateAndGeometryDrift) {
  auto missing = make_fixture();
  missing.observed.pop_back();
  EXPECT_FALSE(QwenInt4SourceBinding::Verify(
      missing.plan, std::move(missing.observed), missing.receipt).ok());

  auto duplicate = make_fixture();
  duplicate.observed.back() = duplicate.observed.front();
  EXPECT_FALSE(QwenInt4SourceBinding::Verify(
      duplicate.plan, std::move(duplicate.observed), duplicate.receipt).ok());

  auto shape = make_fixture();
  shape.observed.front().shape.front() += 1;
  EXPECT_FALSE(QwenInt4SourceBinding::Verify(
      shape.plan, std::move(shape.observed), shape.receipt).ok());
}

TEST(QwenInt4SourceBindingTest, RejectsTiedEvidenceMutation) {
  auto fixture = make_fixture();
  fixture.receipt.lm_head_sha256.bytes[0] ^= std::byte{1};
  EXPECT_FALSE(QwenInt4SourceBinding::Verify(
      fixture.plan, std::move(fixture.observed), fixture.receipt).ok());
}

}  // namespace
}  // namespace pih
