#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/backend/cuda/verified_cubin.h"
#include "pih/core/sha256.h"

namespace pih {
namespace {

std::vector<std::byte> cuda_elf64() {
  std::vector<std::byte> bytes(64);
  bytes[0] = std::byte{0x7f};
  bytes[1] = std::byte{'E'};
  bytes[2] = std::byte{'L'};
  bytes[3] = std::byte{'F'};
  bytes[4] = std::byte{2};
  bytes[5] = std::byte{1};
  bytes[6] = std::byte{1};
  bytes[18] = std::byte{0xbe};
  bytes[20] = std::byte{1};
  return bytes;
}

class VerifiedCubinTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = std::filesystem::temp_directory_path() /
            ("pih-cubin-" + std::to_string(counter_++) + ".cubin");
  }
  void TearDown() override { std::filesystem::remove(path_); }

  void write(std::span<const std::byte> bytes) {
    std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }

  std::filesystem::path path_;
  static inline std::uint64_t counter_ = 1;
};

TEST_F(VerifiedCubinTest, PublishesOwnedBytesAfterExactIdentityCheck) {
  auto source = cuda_elf64();
  source.push_back(std::byte{0x5a});
  write(source);
  const auto expected = sha256(source);
  ASSERT_TRUE(expected.ok());

  auto artifact = VerifiedCubin::Load(path_, source.size(), expected.value());
  ASSERT_TRUE(artifact.ok()) << artifact.status().message();
  EXPECT_EQ(artifact->bytes().size(), source.size());
  EXPECT_EQ(artifact->bytes().back(), std::byte{0x5a});
  EXPECT_EQ(artifact->digest(), expected.value());

  auto from_manifest_text =
      VerifiedCubin::Load(path_, source.size(), expected->hex());
  ASSERT_TRUE(from_manifest_text.ok());

  source.back() = std::byte{0x7b};
  write(source);
  EXPECT_EQ(artifact->bytes().back(), std::byte{0x5a});
}

TEST_F(VerifiedCubinTest, RejectsDigestMismatchAndMalformedElf) {
  auto source = cuda_elf64();
  write(source);
  Sha256Digest wrong{};
  EXPECT_FALSE(VerifiedCubin::Load(path_, source.size(), wrong).ok());

  source[18] = std::byte{0};
  write(source);
  const auto digest = sha256(source);
  ASSERT_TRUE(digest.ok());
  EXPECT_FALSE(VerifiedCubin::Load(path_, source.size(), digest.value()).ok());
}

TEST_F(VerifiedCubinTest, RejectsTruncationAndByteBudgetBeforePublication) {
  const std::array<std::byte, 4> truncated{std::byte{0x7f}, std::byte{'E'},
                                          std::byte{'L'}, std::byte{'F'}};
  write(truncated);
  const auto digest = sha256(truncated);
  ASSERT_TRUE(digest.ok());
  EXPECT_FALSE(VerifiedCubin::Load(path_, 64, digest.value()).ok());

  auto source = cuda_elf64();
  write(source);
  EXPECT_FALSE(VerifiedCubin::Load(path_, 63, sha256(source).value()).ok());
  EXPECT_FALSE(VerifiedCubin::Load(path_, 64, Sha256Digest{}).ok());
  EXPECT_FALSE(VerifiedCubin::Load(path_, 64, "not-a-sha256").ok());
}

}  // namespace
}  // namespace pih
