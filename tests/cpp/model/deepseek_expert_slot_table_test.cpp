#include "pih/model/deepseek_expert_slot_table.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

TEST(DeepSeekExpertSlotTableTest, BindsLeaseIdentityAndGenerationToSlotView) {
  auto table = DeepSeekExpertSlotTable::Create(
      {0x10000000U, 0x11000000U}, 77U, 1U);
  ASSERT_TRUE(table.ok());
  auto bound = table->bind(DeepSeekExpertLease{{3, 9}, 1, 42});
  ASSERT_TRUE(bound.ok());
  EXPECT_EQ(bound->identity, (DeepSeekExpertIdentity{3, 9}));
  EXPECT_EQ(bound->slot, 1U);
  EXPECT_EQ(bound->generation, 42U);
  EXPECT_EQ(bound->context_identity, 77U);
  EXPECT_EQ(bound->device_ordinal, 1U);
  EXPECT_EQ(bound->bundle.w1.packed.address, 0x11000000U);
  EXPECT_EQ(bound->layout_id,
            DeepSeekExpertSlotTable::kCanonicalLayoutId);
}

TEST(DeepSeekExpertSlotTableTest, RejectsOverlappingOrInvalidBacking) {
  EXPECT_FALSE(DeepSeekExpertSlotTable::Create({}, 77U, 0U).ok());
  EXPECT_FALSE(DeepSeekExpertSlotTable::Create(
                   {0x10000000U, 0x10000100U}, 77U, 0U).ok());
  EXPECT_FALSE(DeepSeekExpertSlotTable::Create(
                   {0x10000001U, 0x11000000U}, 77U, 0U).ok());
  EXPECT_FALSE(DeepSeekExpertSlotTable::Create(
                   {0x10000000U, 0x11000000U}, 0U, 0U).ok());
}

TEST(DeepSeekExpertSlotTableTest, RejectsStaleShapedLease) {
  auto table = DeepSeekExpertSlotTable::Create(
      {0x10000000U, 0x11000000U}, 77U, 0U);
  ASSERT_TRUE(table.ok());
  EXPECT_TRUE(table->bind(DeepSeekExpertLease{{42, 0}, 0, 1}).ok());
  EXPECT_FALSE(table->bind(DeepSeekExpertLease{{43, 0}, 0, 1}).ok());
  EXPECT_FALSE(table->bind(DeepSeekExpertLease{{1, 256}, 0, 1}).ok());
  EXPECT_FALSE(table->bind(DeepSeekExpertLease{{1, 1}, 2, 1}).ok());
  EXPECT_FALSE(table->bind(DeepSeekExpertLease{{1, 1}, 0, 0}).ok());
}

TEST(DeepSeekExpertSlotTableTest,
     ResidentOnlyTableBindsOnlyMainBundlesWithoutSlots) {
  auto table = DeepSeekExpertSlotTable::CreateResidentOnly(77U, 0U);
  ASSERT_TRUE(table.ok()) << table.status().message();
  EXPECT_EQ(table->slot_count(), 0U);
  const auto matrix = DeepSeekExpertMatrixDeviceView{
      {0x10000000U, DeepSeekExpertBundleLayout::kPackedBytesPerMatrix},
      {0x20000000U, DeepSeekExpertBundleLayout::kScaleBytesPerMatrix}};
  const DeepSeekExpertBundleDeviceView bundle{matrix, matrix, matrix};
  EXPECT_TRUE(table->bind_resident({3, 7}, 9, bundle).ok());
  EXPECT_TRUE(table->bind_resident({42, 7}, 9, bundle).ok());
  EXPECT_FALSE(table->bind_resident({43, 7}, 9, bundle).ok());
  EXPECT_FALSE(table->bind_resident({2, 7}, 9, bundle).ok());
}

}  // namespace
}  // namespace pih
