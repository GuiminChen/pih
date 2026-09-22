#include "pih/model/deepseek_stage_expert_inventory.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

std::vector<DeepSeekStageExpertExtentRecord> records_for(
    DeepSeekStageRange range) {
  std::vector<DeepSeekStageExpertExtentRecord> records;
  std::uintptr_t address = 0x10000000;
  for (std::uint16_t layer = static_cast<std::uint16_t>(range.first_layer);
       layer <= range.last_layer; ++layer) {
    for (std::uint16_t expert = 0; expert < 256; ++expert) {
      records.push_back({{layer, expert},
                         {address, DeepSeekExpertPager::kBundleBytes,
                          static_cast<std::uint64_t>(layer) + 1U}});
      address += DeepSeekExpertPager::kBundleBytes;
    }
  }
  return records;
}

TEST(DeepSeekStageExpertInventoryTest, ResolvesTotalStageOwnedInventory) {
  auto inventory = DeepSeekStageExpertInventory::Create(
      {11, 12}, records_for({11, 12}));
  ASSERT_TRUE(inventory.ok()) << inventory.status().message();
  EXPECT_EQ(inventory->record_count(), 512U);
  auto first = inventory->resolve({11, 0}, DeepSeekExpertPager::kBundleBytes);
  auto last = inventory->resolve({12, 255}, DeepSeekExpertPager::kBundleBytes);
  ASSERT_TRUE(first.ok()); ASSERT_TRUE(last.ok());
  EXPECT_EQ(first->address, 0x10000000U);
  EXPECT_GT(last->address, first->address);
  EXPECT_EQ(last->registration_identity, 13U);
}

TEST(DeepSeekStageExpertInventoryTest, RejectsMissingOrDuplicateRecord) {
  auto records = records_for({4, 4});
  records.pop_back();
  EXPECT_FALSE(DeepSeekStageExpertInventory::Create({4, 4}, records).ok());
  records = records_for({4, 4});
  records.back().identity = records.front().identity;
  EXPECT_FALSE(DeepSeekStageExpertInventory::Create({4, 4}, records).ok());
}

TEST(DeepSeekStageExpertInventoryTest, RejectsMalformedExtent) {
  auto records = records_for({4, 4});
  records[7].extent.bytes -= 1;
  EXPECT_FALSE(DeepSeekStageExpertInventory::Create({4, 4}, records).ok());
  records = records_for({4, 4});
  records[7].extent.address += 1;
  EXPECT_FALSE(DeepSeekStageExpertInventory::Create({4, 4}, records).ok());
  records = records_for({4, 4});
  records[7].extent.registration_identity = 0;
  EXPECT_FALSE(DeepSeekStageExpertInventory::Create({4, 4}, records).ok());
  records = records_for({4, 4});
  records[7].extent.address = records[6].extent.address;
  EXPECT_FALSE(DeepSeekStageExpertInventory::Create({4, 4}, records).ok());
}

TEST(DeepSeekStageExpertInventoryTest, RejectsForeignLayerAndWrongBytes) {
  auto inventory = DeepSeekStageExpertInventory::Create(
      {4, 4}, records_for({4, 4}));
  ASSERT_TRUE(inventory.ok());
  EXPECT_FALSE(inventory->resolve(
      {3, 1}, DeepSeekExpertPager::kBundleBytes).ok());
  EXPECT_FALSE(inventory->resolve({4, 1}, 1).ok());
}

}  // namespace
}  // namespace pih
