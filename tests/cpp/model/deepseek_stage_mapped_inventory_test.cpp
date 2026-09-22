#include "pih/model/deepseek_stage_mapped_inventory.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekStageMappedInventoryTest, SharesOneStageOwnedMappingInventory) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-stage-inventory-test";
  std::filesystem::create_directory(root);
  std::ofstream(root / "a.safetensors", std::ios::binary | std::ios::trunc)
      << "0123456789";
  std::ofstream(root / "b.safetensors", std::ios::binary | std::ios::trunc)
      << "x";
  auto lease = ControllerFileLease::OpenBeneath(
      root, "a.safetensors", 10,
      ArtifactImmutabilityMode::kUncalibrated).value();
  std::vector<DeepSeekWorkerShardDescriptor> descriptors;
  descriptors.emplace_back("a.safetensors",
                           lease.duplicate_for_worker().value());
  DeepSeekRankMappingPlan plan;
  plan.rank = 2;
  plan.owned_tensor_count = 2;
  plan.logical_tensor_bytes = 6;
  plan.mapped_interval_bytes = 10;
  plan.shards.push_back({"a.safetensors", 10});
  plan.intervals.push_back({"a.safetensors", 0, 10});
  auto extra_lease = ControllerFileLease::OpenBeneath(
      root, "b.safetensors", 1,
      ArtifactImmutabilityMode::kUncalibrated).value();
  std::vector<DeepSeekWorkerShardDescriptor> extra;
  extra.emplace_back("a.safetensors", lease.duplicate_for_worker().value());
  extra.emplace_back("b.safetensors",
                     extra_lease.duplicate_for_worker().value());
  EXPECT_FALSE(
      DeepSeekStageMappedInventory::Create(plan, std::move(extra)).ok());
  auto inventory = DeepSeekStageMappedInventory::Create(
      plan, std::move(descriptors));
  ASSERT_TRUE(inventory.ok()) << inventory.status().message();
  EXPECT_EQ(inventory->rank(), 2U);
  EXPECT_EQ(inventory->mapped_interval_bytes(), 10U);
  auto fixed = inventory->bytes("a.safetensors", 1, 4);
  auto pager = inventory->bytes("a.safetensors", 1, 4);
  ASSERT_TRUE(fixed.ok());
  ASSERT_TRUE(pager.ok());
  EXPECT_EQ(fixed->data(), pager->data());
  EXPECT_EQ(std::to_integer<char>((*fixed)[3]), '4');
  EXPECT_FALSE(inventory->bytes("a.safetensors", 8, 3).ok());
  EXPECT_FALSE(inventory->bytes("unknown", 0, 1).ok());
  std::filesystem::remove_all(root);
}

TEST(DeepSeekStageMappedInventoryTest, RejectsMissingOrExtraDescriptors) {
  DeepSeekRankMappingPlan plan;
  plan.shards.push_back({"a.safetensors", 1});
  plan.intervals.push_back({"a.safetensors", 0, 1});
  EXPECT_FALSE(DeepSeekStageMappedInventory::Create(plan, {}).ok());
}

} }  // namespace pih
