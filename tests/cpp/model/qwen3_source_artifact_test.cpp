#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "pih/core/sha256.h"
#include "pih/model/qwen3_source_artifact.h"
#include "pih/model/safetensors_file.h"

namespace pih {
namespace {

class Qwen3SourceArtifactTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = std::filesystem::temp_directory_path() /
            ("pih-qwen-source-" + std::to_string(counter_++) + ".safetensors");
  }
  void TearDown() override { std::filesystem::remove(path_); }

  void write_file(std::string json, std::string payload) {
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    const auto size = static_cast<std::uint64_t>(json.size());
    for (int index = 0; index < 8; ++index) {
      output.put(static_cast<char>((size >> (index * 8)) & 0xff));
    }
    output.write(json.data(), static_cast<std::streamsize>(json.size()));
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }

  Qwen3SourceArtifactExpectation expectation(const SafetensorsFile& source) {
    return Qwen3SourceArtifactExpectation{
        source.size_bytes(), source.header().tensors().size(),
        source.header().data_bytes(), sha256(source.file_bytes()).value()};
  }

  std::filesystem::path path_;
  static inline std::uint64_t counter_ = 1;
};

TEST_F(Qwen3SourceArtifactTest, ProducesReceiptForPinnedArtifactWithTiedWeights) {
  write_file(
      R"({"model.embed_tokens.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"lm_head.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[4,8]}})",
      "abcdabcd");
  auto source = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(source.ok()) << source.status().message();

  auto receipt = verify_qwen3_source_artifact(*source, expectation(*source));

  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->file_sha256, expectation(*source).file_sha256);
  EXPECT_EQ(receipt->file_bytes, source->size_bytes());
  EXPECT_EQ(receipt->tensor_count, 2);
  EXPECT_EQ(receipt->data_bytes, 8);
  EXPECT_EQ(receipt->tied_weight_bytes, 4);
  EXPECT_NE(receipt->embedding_file_begin, receipt->lm_head_file_begin);
  EXPECT_NE(receipt->embedding_file_end, receipt->lm_head_file_end);
  EXPECT_EQ(receipt->embedding_file_end - receipt->embedding_file_begin, 4U);
  EXPECT_EQ(receipt->lm_head_file_end - receipt->lm_head_file_begin, 4U);
  EXPECT_EQ(receipt->embedding_sha256, receipt->lm_head_sha256);
}

TEST_F(Qwen3SourceArtifactTest, RejectsUnpinnedOrNonTiedArtifact) {
  write_file(
      R"({"model.embed_tokens.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"lm_head.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[4,8]}})",
      "abcdabce");
  auto source = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(source.ok());
  auto expected = expectation(*source);

  expected.file_sha256.bytes[0] ^= std::byte{1};
  auto wrong_digest = verify_qwen3_source_artifact(*source, expected);
  ASSERT_FALSE(wrong_digest.ok());
  EXPECT_EQ(wrong_digest.status().code(), StatusCode::kFailedPrecondition);

  expected = expectation(*source);
  auto untied = verify_qwen3_source_artifact(*source, expected);
  ASSERT_FALSE(untied.ok());
  EXPECT_EQ(untied.status().code(), StatusCode::kFailedPrecondition);
}

TEST_F(Qwen3SourceArtifactTest, RejectsMetadataThatDoesNotMatchSealedManifest) {
  write_file(
      R"({"model.embed_tokens.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"lm_head.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[4,8]}})",
      "abcdabcd");
  auto source = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(source.ok());
  auto expected = expectation(*source);
  ++expected.tensor_count;

  auto receipt = verify_qwen3_source_artifact(*source, expected);

  ASSERT_FALSE(receipt.ok());
  EXPECT_EQ(receipt.status().code(), StatusCode::kFailedPrecondition);
}

TEST_F(Qwen3SourceArtifactTest, RevalidatesSealedReceiptAgainstSameMapping) {
  write_file(
      R"({"model.embed_tokens.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]},"lm_head.weight":{"dtype":"U8","shape":[2,2],"data_offsets":[4,8]}})",
      "abcdabcd");
  auto source = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(source.ok());
  auto receipt = verify_qwen3_source_artifact(*source, expectation(*source));
  ASSERT_TRUE(receipt.ok());
  EXPECT_TRUE(revalidate_qwen3_source_artifact_receipt(*source, *receipt).ok());
  receipt->file_sha256.bytes[0] ^= std::byte{1};
  EXPECT_FALSE(revalidate_qwen3_source_artifact_receipt(*source, *receipt).ok());
}

}  // namespace
}  // namespace pih
