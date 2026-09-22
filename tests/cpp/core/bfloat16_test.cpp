#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "pih/core/bfloat16.h"

namespace pih {
namespace {

float float_from_bits(std::uint32_t bits) { return std::bit_cast<float>(bits); }

TEST(BFloat16Test, ConvertsFiniteValuesAtExactStorageBoundary) {
  EXPECT_EQ(BFloat16::FromFloat(0.0F).bits, 0x0000);
  EXPECT_EQ(BFloat16::FromFloat(-0.0F).bits, 0x8000);
  EXPECT_EQ(BFloat16::FromFloat(1.0F).bits, 0x3f80);
  EXPECT_FLOAT_EQ(BFloat16{0xc020}.to_float(), -2.5F);
  EXPECT_EQ(BFloat16::FromFloat(std::numeric_limits<float>::infinity()).bits,
            0x7f80);
}

TEST(BFloat16Test, RoundsHalfwayValuesToEven) {
  EXPECT_EQ(BFloat16::FromFloat(float_from_bits(0x3f808000)).bits, 0x3f80);
  EXPECT_EQ(BFloat16::FromFloat(float_from_bits(0x3f818000)).bits, 0x3f82);
  EXPECT_EQ(BFloat16::FromFloat(float_from_bits(0xbf808000)).bits, 0xbf80);
}

TEST(BFloat16Test, PreservesNonfiniteClassWithoutProducingInfinityFromNan) {
  for (const std::uint32_t raw : {0x7f800001U, 0x7fc12345U, 0xff800001U}) {
    const BFloat16 converted = BFloat16::FromFloat(float_from_bits(raw));
    EXPECT_TRUE(std::isnan(converted.to_float()));
    EXPECT_EQ(converted.bits & 0x0040U, 0x0040U);
    EXPECT_EQ(converted.bits & 0x8000U, (raw >> 16U) & 0x8000U);
  }
}

TEST(BFloat16Test, HandlesSubnormalAndOverflowBoundariesDeterministically) {
  EXPECT_EQ(BFloat16::FromFloat(float_from_bits(0x00000001)).bits, 0x0000);
  EXPECT_EQ(BFloat16::FromFloat(float_from_bits(0x00008000)).bits, 0x0000);
  EXPECT_EQ(BFloat16::FromFloat(float_from_bits(0x00018000)).bits, 0x0002);
  EXPECT_EQ(BFloat16::FromFloat(float_from_bits(0x7f7fffff)).bits, 0x7f80);
}

}  // namespace
}  // namespace pih
