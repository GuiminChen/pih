#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

TEST(CheckedAddTest, AcceptsExactBoundary) {
  const auto result = checked_add_u64(std::numeric_limits<std::uint64_t>::max() - 1, 1);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), std::numeric_limits<std::uint64_t>::max());
}

TEST(CheckedAddTest, RejectsOverflow) {
  const auto result = checked_add_u64(std::numeric_limits<std::uint64_t>::max(), 1);
  EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
}

TEST(CheckedMultiplyTest, HandlesZeroAndRejectsOverflow) {
  EXPECT_EQ(checked_mul_u64(std::numeric_limits<std::uint64_t>::max(), 0).value(), 0);
  EXPECT_EQ(
      checked_mul_u64(std::numeric_limits<std::uint64_t>::max(), 2).status().code(),
      StatusCode::kResourceExhausted);
}

TEST(CheckedAlignTest, AlignsWithoutOverflow) {
  EXPECT_EQ(checked_align_up_u64(0, 8).value(), 0);
  EXPECT_EQ(checked_align_up_u64(9, 8).value(), 16);
}

TEST(CheckedAlignTest, RejectsInvalidAlignmentAndOverflow) {
  EXPECT_EQ(checked_align_up_u64(1, 0).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(checked_align_up_u64(1, 3).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(
      checked_align_up_u64(std::numeric_limits<std::uint64_t>::max(), 8).status().code(),
      StatusCode::kResourceExhausted);
}

}  // namespace
}  // namespace pih
