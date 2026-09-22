#include "pih/platform/linux/linux_deepseek_rank_artifact_prefault_operations.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>

#include "pih/io/controller_file_lease.h"
#include "pih/io/descriptor_mapped_range.h"

namespace pih {
namespace {

TEST(LinuxDeepSeekRankArtifactPrefaultOperationsTest,
     TouchesAndProvesEverySelectedBasePage) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("pih-linux-prefault-" +
                     std::to_string(static_cast<std::uint64_t>(::getpid())));
  std::filesystem::remove_all(root);
  ASSERT_TRUE(std::filesystem::create_directory(root));
  constexpr std::uint64_t kFileBytes = 8193;
  {
    std::ofstream output(root / "weights.bin",
                         std::ios::binary | std::ios::trunc);
    output.seekp(static_cast<std::streamoff>(kFileBytes - 1U));
    output.put('\0');
  }
  auto lease = ControllerFileLease::OpenBeneath(
      root, "weights.bin", kFileBytes,
      ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(lease.ok()) << lease.status().message();
  auto descriptor = lease->duplicate_for_worker();
  ASSERT_TRUE(descriptor.ok()) << descriptor.status().message();
  auto mapping = DescriptorMappedRange::MapReadOnly(
      *descriptor, 0, kFileBytes);
  ASSERT_TRUE(mapping.ok()) << mapping.status().message();
  auto operations =
      LinuxDeepSeekRankArtifactPrefaultOperations::Create();
  ASSERT_TRUE(operations.ok()) << operations.status().message();
  auto before = (*operations)->sample_resources();
  ASSERT_TRUE(before.ok()) << before.status().message();
  auto observed = (*operations)->prefault_range(
      std::span<const std::byte>(mapping->data(),
                                 static_cast<std::size_t>(kFileBytes)),
      4096);
  ASSERT_TRUE(observed.ok()) << observed.status().message();
  EXPECT_LE(observed->resident_page_bytes_before, 12288U);
  EXPECT_EQ(observed->resident_page_bytes_after, 12288U);
  EXPECT_EQ(observed->touched_page_count, 3U);
  auto after = (*operations)->sample_resources();
  ASSERT_TRUE(after.ok()) << after.status().message();
  EXPECT_GE(after->monotonic_ns, before->monotonic_ns);
  EXPECT_GE(after->major_fault_count, before->major_fault_count);
  EXPECT_FALSE((*operations)->prefault_range(
                   std::span<const std::byte>(mapping->data() + 1, 1),
                   4096)
                   .ok());
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace pih
