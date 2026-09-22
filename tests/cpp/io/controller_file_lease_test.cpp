#include "pih/io/controller_file_lease.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace pih { namespace {

class ControllerFileLeaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("pih-controller-lease-" + std::to_string(counter_++));
    std::filesystem::create_directory(root_);
    std::ofstream(root_ / "shard.bin", std::ios::binary) << "0123456789";
  }
  void TearDown() override { std::filesystem::remove_all(root_); }

  std::filesystem::path root_;
  static inline std::uint64_t counter_ = 1;
};

TEST_F(ControllerFileLeaseTest, ReadsAndDuplicatesTheSameOpenedIdentity) {
  auto lease = ControllerFileLease::OpenBeneath(
      root_, "shard.bin", 10, ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(lease.ok()) << lease.status().message();
  EXPECT_EQ(lease->identity().file_bytes, 10U);
  EXPECT_FALSE(lease->production_eligible());

  std::array<std::byte, 4> bytes{};
  ASSERT_TRUE(lease->read_exact(3, bytes).ok());
  EXPECT_EQ(std::to_integer<char>(bytes[0]), '3');
  EXPECT_EQ(std::to_integer<char>(bytes[3]), '6');

  auto worker = lease->duplicate_for_worker();
  ASSERT_TRUE(worker.ok());
  EXPECT_EQ(worker->identity(), lease->identity());
  ASSERT_TRUE(worker->read_exact(8, std::span<std::byte>(bytes).first(2)).ok());
  EXPECT_EQ(std::to_integer<char>(bytes[1]), '9');
  EXPECT_TRUE(lease->poll_identity_unchanged().ok());
}

TEST_F(ControllerFileLeaseTest, PollDetectsInPlaceDescriptorMutation) {
  auto lease = ControllerFileLease::OpenBeneath(
      root_, "shard.bin", 10, ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(lease.ok());
  std::ofstream(root_ / "shard.bin", std::ios::binary | std::ios::trunc)
      << "short";
  EXPECT_FALSE(lease->poll_identity_unchanged().ok());
}

TEST_F(ControllerFileLeaseTest, PollIgnoresAtomicPathReplacement) {
  auto lease = ControllerFileLease::OpenBeneath(
      root_, "shard.bin", 10, ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(lease.ok());
  const auto old_path = root_ / "old-shard.bin";
  std::filesystem::rename(root_ / "shard.bin", old_path);
  std::ofstream(root_ / "shard.bin", std::ios::binary) << "replacement";
  EXPECT_TRUE(lease->poll_identity_unchanged().ok());
  std::filesystem::remove(old_path);
}

TEST_F(ControllerFileLeaseTest, RejectsAuthorityEscapeAndInvalidBounds) {
  EXPECT_FALSE(ControllerFileLease::OpenBeneath(
                   root_, "../shard.bin", 10,
                   ArtifactImmutabilityMode::kUncalibrated)
                   .ok());
  EXPECT_FALSE(ControllerFileLease::OpenBeneath(
                   root_, "sub/shard.bin", 10,
                   ArtifactImmutabilityMode::kUncalibrated)
                   .ok());
  EXPECT_FALSE(ControllerFileLease::OpenBeneath(
                   root_, "C:shard.bin", 10,
                   ArtifactImmutabilityMode::kUncalibrated)
                   .ok());
  EXPECT_FALSE(ControllerFileLease::OpenBeneath(
                   root_, "shard.bin", 9,
                   ArtifactImmutabilityMode::kUncalibrated)
                   .ok());

  auto lease = ControllerFileLease::OpenBeneath(
      root_, "shard.bin", 10, ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(lease.ok());
  std::array<std::byte, 3> bytes{};
  EXPECT_FALSE(lease->read_exact(8, bytes).ok());
}

TEST_F(ControllerFileLeaseTest, CannotClaimProductionWithoutVerityReceipt) {
  EXPECT_FALSE(ControllerFileLease::OpenBeneath(
                   root_, "shard.bin", 10,
                   ArtifactImmutabilityMode::kFsVerity)
                   .ok());
  EXPECT_FALSE(ControllerFileLease::OpenBeneath(
                   root_, "shard.bin", 10,
                   ArtifactImmutabilityMode::kDmVeritySnapshot)
                   .ok());
}

TEST_F(ControllerFileLeaseTest, ValidatesMeasuredFsVerityDigestExactly) {
  Sha256Digest expected{};
  expected.bytes[0] = std::byte{0x42};
  EXPECT_TRUE(validate_fsverity_sha256_measurement(
                  expected, 1, std::span<const std::byte>(expected.bytes))
                  .ok());
  auto wrong = expected;
  wrong.bytes[31] = std::byte{1};
  EXPECT_FALSE(validate_fsverity_sha256_measurement(
                   expected, 1, std::span<const std::byte>(wrong.bytes))
                   .ok());
  EXPECT_FALSE(validate_fsverity_sha256_measurement(
                   expected, 2, std::span<const std::byte>(expected.bytes))
                   .ok());
  EXPECT_FALSE(validate_fsverity_sha256_measurement(
                   expected, 1,
                   std::span<const std::byte>(expected.bytes).first(31))
                   .ok());
}

#ifdef _WIN32
TEST_F(ControllerFileLeaseTest, FsVerityProductionOpenFailsClosedOffLinux) {
  Sha256Digest expected{};
  EXPECT_FALSE(ControllerFileLease::OpenBeneathFsVerity(
                   root_, "shard.bin", 10, expected)
                   .ok());
}

TEST_F(ControllerFileLeaseTest, DmVerityProductionOpenFailsClosedOffLinux) {
  DmVeritySupervisorReceipt receipt;
  Sha256Digest expected{};
  EXPECT_FALSE(ControllerFileLease::OpenBeneathDmVeritySnapshot(
                   root_, "shard.bin", 10, receipt, expected, expected,
                   expected, 1536ULL * 1024 * 1024)
                   .ok());
}
#endif

} }  // namespace pih
