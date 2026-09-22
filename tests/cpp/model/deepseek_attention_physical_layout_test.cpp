#include "pih/model/deepseek_attention_physical_layout.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(DeepSeekAttentionPhysicalLayoutTest,
     CompilesAlignedNonOverlappingVersionedExtents) {
  const DeepSeekStagePlan stage{0, {2, 3}, false, false, false};
  auto layout = DeepSeekAttentionPhysicalLayout::Compile(stage, 2, 256);
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_EQ(layout->version(), 1U);
  EXPECT_EQ(layout->extents().size(), 3U);
  EXPECT_EQ(layout->ratio4_page_pairs_per_sequence(), 2U);
  EXPECT_EQ(layout->ratio128_pages_per_sequence(), 2U);
  EXPECT_EQ(layout->ratio4_pair_count(), 4U);
  EXPECT_EQ(layout->ratio128_page_count(), 4U);
  EXPECT_EQ(layout->identity().hex().size(), 64U);

  std::uint64_t preceding_end = 0;
  for (const auto& extent : layout->extents()) {
    EXPECT_EQ(extent.offset_bytes % extent.alignment_bytes, 0U);
    EXPECT_GE(extent.offset_bytes, preceding_end);
    EXPECT_EQ(extent.bytes, extent.slot_count * extent.page_bytes);
    preceding_end = extent.offset_bytes + extent.bytes;
  }
  EXPECT_EQ(layout->allocation_bytes(), preceding_end);
  EXPECT_EQ(layout->extent(DeepSeekAttentionPhysicalExtentKind::kRatio4Main)
                .slot_count,
            4U);
}

TEST(DeepSeekAttentionPhysicalLayoutTest,
     RejectsInvalidStageCapacityAndSlotOverflow) {
  EXPECT_FALSE(DeepSeekAttentionPhysicalLayout::Compile(
      {0, {0, 1}, true, false, false}, 1, 256).ok());
  EXPECT_FALSE(DeepSeekAttentionPhysicalLayout::Compile(
      {0, {2, 3}, false, false, false}, 0, 256).ok());
  EXPECT_FALSE(DeepSeekAttentionPhysicalLayout::Compile(
      {0, {0, 42}, false, false, false}, UINT32_MAX, 1048576).ok());
}

TEST(DeepSeekAttentionPhysicalLayoutTest,
     IdentityBindsCapacityAndStageOwnership) {
  const DeepSeekStagePlan stage{0, {2, 3}, false, false, false};
  auto first = DeepSeekAttentionPhysicalLayout::Compile(stage, 2, 256);
  auto capacity = DeepSeekAttentionPhysicalLayout::Compile(stage, 3, 256);
  auto context = DeepSeekAttentionPhysicalLayout::Compile(stage, 2, 512);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(capacity.ok());
  ASSERT_TRUE(context.ok());
  EXPECT_NE(first->identity(), capacity->identity());
  EXPECT_NE(first->identity(), context->identity());
}

}  // namespace
}  // namespace pih
