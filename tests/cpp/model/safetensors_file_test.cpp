#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/safetensors_file.h"
#include "pih/io/controller_file_lease.h"

namespace pih {
namespace {

class SafetensorsFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = std::filesystem::temp_directory_path() /
            ("pih-safetensors-file-" + std::to_string(counter_++) + ".bin");
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

  std::filesystem::path path_;
  static inline std::uint64_t counter_ = 1;
};

TEST_F(SafetensorsFileTest, OwnsMappingAndReturnsVerifiedImmutableTensorBytes) {
  write_file(
      R"({"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})",
      "abcd");
  auto opened = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(opened.ok()) << opened.status().message();
  EXPECT_EQ(opened->header().tensors().size(), 1);
  auto tensor = opened->tensor("weight");
  ASSERT_TRUE(tensor.ok());
  EXPECT_EQ(tensor->dtype, DType::kUInt8);
  ASSERT_EQ(tensor->shape.size(), 1);
  EXPECT_EQ(tensor->shape[0], 4);
  ASSERT_EQ(tensor->bytes.size(), 4);
  EXPECT_EQ(std::to_integer<char>(tensor->bytes[2]), 'c');
}

TEST_F(SafetensorsFileTest, RejectsUnknownTensorAndMalformedFileAtomically) {
  write_file(
      R"({"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})",
      "abcd");
  auto opened = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(opened.ok());
  EXPECT_FALSE(opened->tensor("missing").ok());

  opened = Status::Internal("release mapping before rewrite");
  write_file(
      R"({"weight":{"dtype":"U8","shape":[5],"data_offsets":[0,4]}})",
      "abcd");
  EXPECT_FALSE(SafetensorsFile::Open(path_, 1024).ok());
}

TEST_F(SafetensorsFileTest, MovePreservesCatalogAndClearsSource) {
  write_file(
      R"({"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
      "z");
  SafetensorsFile first = std::move(SafetensorsFile::Open(path_, 1024)).value();
  SafetensorsFile second = std::move(first);
  EXPECT_EQ(first.size_bytes(), 0);
  ASSERT_TRUE(second.tensor("weight").ok());
}

TEST_F(SafetensorsFileTest, ControllerHeaderReceiptDoesNotRequireDataMapping) {
  const std::string json =
      R"({"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";
  write_file(json, "abcd");
  auto receipt = load_safetensors_header_file(path_, 1024);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->file_bytes, 8U + json.size() + 4U);
  EXPECT_EQ(receipt->prefix_bytes, 8U + json.size());
  ASSERT_NE(receipt->header.tensor("weight"), nullptr);
  EXPECT_NE(receipt->prefix_sha256, Sha256Digest{});
  EXPECT_FALSE(load_safetensors_header_file(path_, 8).ok());
}

TEST_F(SafetensorsFileTest, ControllerHeaderReceiptUsesMasterLeaseNotPath) {
  const std::string old_json =
      R"({"old.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";
  write_file(old_json, "abcd");
  auto lease = ControllerFileLease::OpenBeneath(
      path_.parent_path(), path_.filename().string(), 1024,
      ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(lease.ok());
  const auto replaced = path_.string() + ".old";
  std::filesystem::rename(path_, replaced);
  write_file(
      R"({"new.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})",
      "wxyz");
  auto receipt = load_safetensors_header_descriptor(*lease);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_NE(receipt->header.tensor("old.weight"), nullptr);
  EXPECT_EQ(receipt->header.tensor("new.weight"), nullptr);
  std::filesystem::remove(replaced);
}

}  // namespace
}  // namespace pih
