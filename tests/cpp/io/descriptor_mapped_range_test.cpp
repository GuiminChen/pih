#include "pih/io/descriptor_mapped_range.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "pih/io/controller_file_lease.h"

namespace pih { namespace {

TEST(DescriptorMappedRangeTest, MapsOnlyRequestedDescriptorRange) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-descriptor-map-test";
  std::filesystem::create_directory(root);
  std::ofstream(root / "shard.bin", std::ios::binary | std::ios::trunc)
      << "0123456789";
  auto lease = ControllerFileLease::OpenBeneath(
      root, "shard.bin", 10, ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(lease.ok());
  auto worker = lease->duplicate_for_worker();
  ASSERT_TRUE(worker.ok());
  auto mapping = DescriptorMappedRange::MapReadOnly(*worker, 3, 4);
  ASSERT_TRUE(mapping.ok()) << mapping.status().message();
  EXPECT_EQ(mapping->file_offset(), 3U);
  EXPECT_EQ(mapping->size_bytes(), 4U);
  EXPECT_EQ(std::to_integer<char>(mapping->data()[0]), '3');
  EXPECT_EQ(std::to_integer<char>(mapping->data()[3]), '6');
  EXPECT_FALSE(mapping->slice(3, 2).ok());
  std::filesystem::remove_all(root);
}

TEST(DescriptorMappedRangeTest, MappingOutlivesWorkerDescriptor) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-descriptor-map-lifetime-test";
  std::filesystem::create_directory(root);
  std::ofstream(root / "shard.bin", std::ios::binary | std::ios::trunc)
      << "abcdefghij";
  auto lease = ControllerFileLease::OpenBeneath(
      root, "shard.bin", 10, ArtifactImmutabilityMode::kUncalibrated).value();
  auto worker = lease.duplicate_for_worker().value();
  auto mapping = DescriptorMappedRange::MapReadOnly(worker, 5, 5).value();
  worker = lease.duplicate_for_worker().value();
  EXPECT_EQ(std::to_integer<char>(mapping.data()[0]), 'f');
  EXPECT_FALSE(DescriptorMappedRange::MapReadOnly(worker, 0, 0).ok());
  EXPECT_FALSE(DescriptorMappedRange::MapReadOnly(worker, 9, 2).ok());
  std::filesystem::remove_all(root);
}

} }  // namespace pih
