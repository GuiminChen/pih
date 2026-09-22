#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "pih/backend/cuda/elementwise_launch.h"

namespace pih {
namespace {

TEST(ElementwiseLaunchTest, ComputesExactBoundedCoverage) {
  auto exact = ElementwiseLaunch::Create(256, 256);
  ASSERT_TRUE(exact.ok());
  EXPECT_EQ(exact->grid_blocks(), 1);
  EXPECT_EQ(exact->covered_threads(), 256);

  auto tail = ElementwiseLaunch::Create(257, 256);
  ASSERT_TRUE(tail.ok());
  EXPECT_EQ(tail->grid_blocks(), 2);
  EXPECT_EQ(tail->covered_threads(), 512);
  EXPECT_EQ(tail->elements(), 257);
}

TEST(ElementwiseLaunchTest, RejectsZeroAndNonWarpOrOversizedBlocks) {
  EXPECT_FALSE(ElementwiseLaunch::Create(0, 256).ok());
  EXPECT_FALSE(ElementwiseLaunch::Create(1, 0).ok());
  EXPECT_FALSE(ElementwiseLaunch::Create(1, 31).ok());
  EXPECT_FALSE(ElementwiseLaunch::Create(1, 33).ok());
  EXPECT_FALSE(ElementwiseLaunch::Create(1, 1056).ok());
}

TEST(ElementwiseLaunchTest, RejectsGridOverflowAndCeilingAdditionOverflow) {
  const std::uint64_t maximum =
      static_cast<std::uint64_t>(ElementwiseLaunch::kMaximumGridBlocks) * 32U;
  EXPECT_TRUE(ElementwiseLaunch::Create(maximum, 32).ok());
  EXPECT_FALSE(ElementwiseLaunch::Create(maximum + 1, 32).ok());
  EXPECT_FALSE(
      ElementwiseLaunch::Create(std::numeric_limits<std::uint64_t>::max(), 32).ok());
}

}  // namespace
}  // namespace pih
