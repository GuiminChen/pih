#include "pih/io/exclusive_file_publisher.h"

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace pih {
namespace {

class ExclusiveFilePublisherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto base = "pih-publish-" + std::to_string(counter_++);
    staging_ = std::filesystem::temp_directory_path() / (base + ".staging");
    published_ = std::filesystem::temp_directory_path() / (base + ".artifact");
    std::filesystem::remove(staging_);
    std::filesystem::remove(published_);
  }
  void TearDown() override {
    std::filesystem::remove(staging_);
    std::filesystem::remove(published_);
  }
  void write(const std::filesystem::path& path, std::string_view value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
  }
  std::filesystem::path staging_;
  std::filesystem::path published_;
  static inline std::uint64_t counter_ = 1;
};

TEST_F(ExclusiveFilePublisherTest, PublishesVerifiedGenerationWithoutOverwrite) {
  write(staging_, "artifact");
  const auto digest = sha256(std::as_bytes(std::span("artifact", 8))).value();
  auto receipt = publish_verified_file_exclusive(staging_, published_, 8, digest);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_FALSE(std::filesystem::exists(staging_));
  EXPECT_TRUE(std::filesystem::exists(published_));
  write(staging_, "artifact");
  EXPECT_FALSE(publish_verified_file_exclusive(staging_, published_, 8, digest).ok());
  EXPECT_TRUE(std::filesystem::exists(staging_));
}

TEST_F(ExclusiveFilePublisherTest, PreservesStagingWhenEvidenceMismatches) {
  write(staging_, "artifact");
  auto digest = sha256(std::as_bytes(std::span("artifact", 8))).value();
  digest.bytes[0] ^= std::byte{1};
  EXPECT_FALSE(publish_verified_file_exclusive(staging_, published_, 8, digest).ok());
  EXPECT_TRUE(std::filesystem::exists(staging_));
  EXPECT_FALSE(std::filesystem::exists(published_));
}

}  // namespace
}  // namespace pih
