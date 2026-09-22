#include "pih/model/deepseek_expert_bundle_layout.h"

#include <gtest/gtest.h>

#include <limits>

namespace pih {
namespace {

TEST(DeepSeekExpertBundleLayoutTest, FreezesCanonicalSixSegmentOrder) {
  auto view = DeepSeekExpertBundleLayout::Bind(0x10000000U);
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view->w1.packed.address, 0x10000000U);
  EXPECT_EQ(view->w1.packed.bytes, 4194304U);
  EXPECT_EQ(view->w1.scales.address, 0x10400000U);
  EXPECT_EQ(view->w1.scales.bytes, 262144U);
  EXPECT_EQ(view->w2.packed.address, 0x10440000U);
  EXPECT_EQ(view->w2.scales.address, 0x10840000U);
  EXPECT_EQ(view->w3.packed.address, 0x10880000U);
  EXPECT_EQ(view->w3.scales.address, 0x10C80000U);
  EXPECT_EQ(view->w3.scales.address + view->w3.scales.bytes,
            0x10000000U + DeepSeekExpertBundleLayout::kBundleBytes);
}

TEST(DeepSeekExpertBundleLayoutTest, DerivesOfficialMatrixGeometry) {
  EXPECT_EQ(DeepSeekExpertBundleLayout::kHiddenSize, 4096U);
  EXPECT_EQ(DeepSeekExpertBundleLayout::kIntermediateSize, 2048U);
  EXPECT_EQ(DeepSeekExpertBundleLayout::kPackedBytesPerMatrix, 4194304U);
  EXPECT_EQ(DeepSeekExpertBundleLayout::kScaleBytesPerMatrix, 262144U);
  EXPECT_EQ(DeepSeekExpertBundleLayout::kBundleBytes, 13369344U);
}

TEST(DeepSeekExpertBundleLayoutTest, RejectsUnalignedAndOverflowingBase) {
  EXPECT_FALSE(DeepSeekExpertBundleLayout::Bind(0).ok());
  EXPECT_FALSE(DeepSeekExpertBundleLayout::Bind(0x10000001U).ok());
  const auto overflowing = std::numeric_limits<std::uintptr_t>::max() - 255U;
  const auto alignment_mask =
      ~static_cast<std::uintptr_t>(DeepSeekExpertBundleLayout::kAlignment - 1U);
  EXPECT_FALSE(
      DeepSeekExpertBundleLayout::Bind(overflowing & alignment_mask).ok());
}

}  // namespace
}  // namespace pih
