#include "pih/model/qwen3_bf16_packed_staging_resources.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenBf16PackedStagingResourcesTest,
     BindsAllFourteenByteExactSlotsToOneOwnerGeneration) {
  const auto layout =
      QwenBf16PackedStepStagingLayout::CreateBounded(5, 2, 4).value();
  const QwenBf16DeviceArenaOwner owner{
      0x100000, layout.total_bytes(), 41};
  auto resources = QwenBf16PackedStagingResources::Create(
      41, 0, layout, owner);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(resources->size(), 14U);
  EXPECT_EQ(resources->request_generation(), 41U);
  EXPECT_EQ(resources->owning_rank(), 0);
  for (std::size_t i = 0; i < resources->size(); ++i) {
    const auto slot = static_cast<QwenBf16PackedStagingSlot>(i);
    auto view = resources->view(slot);
    ASSERT_TRUE(view.ok()) << view.status().message();
    EXPECT_EQ(view->dtype(), DType::kUInt8);
    EXPECT_EQ(view->rank(), 1U);
    EXPECT_EQ(view->generation(), 41U);
    EXPECT_EQ(view->byte_span(), layout.copy_spans()[i].size_bytes);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(view->data()),
              owner.base + layout.copy_spans()[i].offset_bytes);
  }
}

TEST(QwenBf16PackedStagingResourcesTest,
     RejectsShortStaleAndCrossRequestOwners) {
  const auto layout =
      QwenBf16PackedStepStagingLayout::CreateBounded(5, 2, 4).value();
  EXPECT_FALSE(QwenBf16PackedStagingResources::Create(
                   41, 0, layout,
                   {0x100000, layout.total_bytes() - 1, 41})
                   .ok());
  EXPECT_FALSE(QwenBf16PackedStagingResources::Create(
                   41, 0, layout,
                   {0x100000, layout.total_bytes(), 0})
                   .ok());
  EXPECT_FALSE(QwenBf16PackedStagingResources::Create(
                   41, 0, layout,
                   {0x100000, layout.total_bytes(), 42})
                   .ok());
  EXPECT_FALSE(QwenBf16PackedStagingResources::Create(
                   0, 0, layout,
                   {0x100000, layout.total_bytes(), 41})
                   .ok());
  EXPECT_FALSE(QwenBf16PackedStagingResources::Create(
                   41, -1, layout,
                   {0x100000, layout.total_bytes(), 41})
                   .ok());
}

TEST(QwenBf16PackedStagingResourcesTest, RejectsUnknownSlot) {
  const auto layout =
      QwenBf16PackedStepStagingLayout::CreateBounded(5, 2, 4).value();
  auto resources = QwenBf16PackedStagingResources::Create(
                       41, 0, layout,
                       {0x100000, layout.total_bytes(), 41})
                       .value();
  EXPECT_FALSE(resources.view(
      static_cast<QwenBf16PackedStagingSlot>(255)).ok());
}

}  // namespace
}  // namespace pih
