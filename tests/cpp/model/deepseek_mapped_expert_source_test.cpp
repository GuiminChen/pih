#include "pih/model/deepseek_mapped_expert_source.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

namespace pih { namespace {

std::uintptr_t aligned_address(std::vector<std::byte>& storage) {
  const auto raw = reinterpret_cast<std::uintptr_t>(storage.data());
  return (raw + 255U) & ~static_cast<std::uintptr_t>(255U);
}

TEST(DeepSeekMappedExpertSourceTest, GathersCanonicalBundleIntoBoundedPool) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-mapped-expert-source-test";
  std::filesystem::create_directory(root);
  const auto shard = root / "expert.safetensors";
  {
    std::ofstream output(shard, std::ios::binary | std::ios::trunc);
    output.seekp(static_cast<std::streamoff>(
                     DeepSeekExpertBundleLayout::kBundleBytes - 1U));
    output.put('\0');
    const std::array<std::uint64_t, 6> offsets = {
        0, 4194304, 4456448, 8650752, 8912896, 13107200};
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      output.seekp(static_cast<std::streamoff>(offsets[i]));
      output.put(static_cast<char>(i + 1U));
    }
  }
  auto lease = ControllerFileLease::OpenBeneath(
      root, "expert.safetensors", DeepSeekExpertBundleLayout::kBundleBytes,
      ArtifactImmutabilityMode::kUncalibrated).value();
  std::vector<DeepSeekWorkerShardDescriptor> descriptors;
  descriptors.emplace_back("expert.safetensors",
                           lease.duplicate_for_worker().value());
  DeepSeekRankMappingPlan plan;
  plan.shards.push_back(
      {"expert.safetensors", DeepSeekExpertBundleLayout::kBundleBytes});
  plan.intervals.push_back(
      {"expert.safetensors", 0, DeepSeekExpertBundleLayout::kBundleBytes});
  plan.mapped_interval_bytes = DeepSeekExpertBundleLayout::kBundleBytes;
  auto inventory = DeepSeekStageMappedInventory::Create(plan, std::move(descriptors));
  ASSERT_TRUE(inventory.ok()) << inventory.status().message();

  const std::array<std::uint64_t, 6> offsets = {
      0, 4194304, 4456448, 8650752, 8912896, 13107200};
  const std::array<std::uint64_t, 6> sizes = {
      4194304, 262144, 4194304, 262144, 4194304, 262144};
  std::vector<DeepSeekExpertMappedBundleRecord> records;
  for (std::uint16_t expert = 0; expert < 256; ++expert) {
    DeepSeekExpertMappedBundleRecord record;
    record.identity = {4, expert};
    for (std::size_t i = 0; i < 6; ++i) {
      record.segments[i] = {"expert.safetensors", offsets[i], sizes[i]};
    }
    records.push_back(std::move(record));
  }
  std::vector<std::byte> first(DeepSeekExpertBundleLayout::kBundleBytes + 255U);
  std::vector<std::byte> second(DeepSeekExpertBundleLayout::kBundleBytes + 255U);
  EXPECT_FALSE(DeepSeekMappedExpertSource::Create(
      *inventory, {4, 4}, records,
      {{aligned_address(first), DeepSeekExpertBundleLayout::kBundleBytes, 1}}).ok());
  EXPECT_FALSE(DeepSeekMappedExpertSource::Create(
      *inventory, {4, 4}, {},
      {{aligned_address(first), DeepSeekExpertBundleLayout::kBundleBytes, 1},
       {aligned_address(second), DeepSeekExpertBundleLayout::kBundleBytes, 2}}).ok());
  auto source = DeepSeekMappedExpertSource::Create(
      *inventory, {4, 4}, std::move(records),
      {{aligned_address(first), DeepSeekExpertBundleLayout::kBundleBytes, 1},
       {aligned_address(second), DeepSeekExpertBundleLayout::kBundleBytes, 2}});
  ASSERT_TRUE(source.ok()) << source.status().message();
  auto one = source->resolve({4, 7}, DeepSeekExpertBundleLayout::kBundleBytes);
  auto two = source->resolve({4, 8}, DeepSeekExpertBundleLayout::kBundleBytes);
  ASSERT_TRUE(one.ok());
  ASSERT_TRUE(two.ok());
  EXPECT_EQ(source->resolve({4, 9}, DeepSeekExpertBundleLayout::kBundleBytes)
                .status().code(), StatusCode::kResourceExhausted);
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_EQ(*reinterpret_cast<const unsigned char*>(one->address + offsets[i]),
              i + 1U);
  }
  EXPECT_FALSE(source->release({4, 8}, *one).ok());
  EXPECT_TRUE(source->release({4, 7}, *one).ok());
  EXPECT_TRUE(source->resolve({4, 9}, DeepSeekExpertBundleLayout::kBundleBytes).ok());
  EXPECT_TRUE(source->release({4, 8}, *two).ok());
  std::filesystem::remove_all(root);
}

} }  // namespace pih
