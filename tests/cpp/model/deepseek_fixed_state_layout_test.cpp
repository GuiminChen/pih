#include "pih/model/deepseek_fixed_state_layout.h"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace pih { namespace {

TEST(DeepSeekFixedStateLayoutTest, FreezesFullMainBankBytesAndAlignment) {
  std::vector<std::uint32_t> layers(43);
  std::iota(layers.begin(), layers.end(), 0U);
  auto layout = DeepSeekFixedStateLayout::Build(layers, false);
  ASSERT_TRUE(layout.ok());
  EXPECT_EQ(layout->total_bytes(), 17842176U);
  EXPECT_EQ(layout->descriptors().size(), 43U);
  for (const auto& descriptor : layout->descriptors()) {
    EXPECT_EQ(descriptor.recent_bf16.offset % 256U, 0U);
    EXPECT_EQ(descriptor.main_kv_state_f32.offset % 256U, 0U);
    EXPECT_EQ(descriptor.main_score_state_f32.offset % 256U, 0U);
  }
}

TEST(DeepSeekFixedStateLayoutTest, ResolvesTypedRatio4AndRatio128Views) {
  const std::vector<std::uint32_t> layers{2, 3};
  auto layout = DeepSeekFixedStateLayout::Build(layers, true).value();
  auto ratio4 = layout.Resolve(2, {0x100000, layout.total_bytes()});
  auto ratio128 = layout.Resolve(3, {0x100000, layout.total_bytes()});
  auto dspark = layout.ResolveDspark(
      DeepSeekDsparkStageId::kMtp2,
      {0x100000, layout.total_bytes()});
  ASSERT_TRUE(ratio4.ok() && ratio128.ok() && dspark.ok());
  EXPECT_EQ(ratio4->main_kv_state_f32.bytes, 32768U);
  EXPECT_EQ(ratio4->index_score_state_f32.bytes, 8192U);
  EXPECT_EQ(ratio128->main_kv_state_f32.bytes, 262144U);
  EXPECT_EQ(ratio128->index_kv_state_f32.address, 0U);
  EXPECT_EQ(dspark->recent_bf16.bytes, 131072U);
  EXPECT_EQ(dspark->stage, DeepSeekDsparkStageId::kMtp2);
  EXPECT_EQ(layout.descriptors().size(), 2U);
  EXPECT_EQ(layout.dspark_descriptors().size(), 3U);
  EXPECT_FALSE(layout.Resolve(43, {0x100000, layout.total_bytes()}).ok());
}

TEST(DeepSeekFixedStateLayoutTest, RejectsUnownedDuplicateAndShortBank) {
  const std::vector<std::uint32_t> duplicate{2, 2};
  EXPECT_FALSE(DeepSeekFixedStateLayout::Build(duplicate, false).ok());
  const std::vector<std::uint32_t> layers{2};
  auto layout = DeepSeekFixedStateLayout::Build(layers, false).value();
  EXPECT_FALSE(layout.Resolve(3, {0x1000, layout.total_bytes()}).ok());
  EXPECT_FALSE(layout.Resolve(2, {0x1000, layout.total_bytes() - 1}).ok());
}

} }  // namespace pih
