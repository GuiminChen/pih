#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "pih/core/float16.h"

namespace pih {

float from_bits(std::uint32_t bits) { return std::bit_cast<float>(bits); }

TEST(Float16Test, ConvertsFiniteAndNonfiniteClasses) {
  EXPECT_EQ(Float16::FromFloat(0.0F).bits, 0x0000);
  EXPECT_EQ(Float16::FromFloat(-0.0F).bits, 0x8000);
  EXPECT_EQ(Float16::FromFloat(1.0F).bits, 0x3c00);
  EXPECT_FLOAT_EQ(Float16{0xc100}.to_float(), -2.5F);
  EXPECT_EQ(Float16::FromFloat(std::numeric_limits<float>::infinity()).bits,
            0x7c00);
  EXPECT_TRUE(std::isnan(Float16{0x7e00}.to_float()));
}

TEST(Float16Test, UsesTiesToEvenAcrossNormalAndSubnormalBoundaries) {
  EXPECT_EQ(Float16::FromFloat(from_bits(0x3f801000)).bits, 0x3c00);
  EXPECT_EQ(Float16::FromFloat(from_bits(0x3f803000)).bits, 0x3c02);
  EXPECT_EQ(Float16::FromFloat(std::ldexp(1.0F, -25)).bits, 0x0000);
  EXPECT_EQ(Float16::FromFloat(std::ldexp(3.0F, -25)).bits, 0x0002);
  EXPECT_EQ(Float16{0x0001}.to_float(), std::ldexp(1.0F, -24));
}

}  // namespace pih
