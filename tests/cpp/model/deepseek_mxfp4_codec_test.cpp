#include "pih/model/deepseek_mxfp4_codec.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

namespace pih {
namespace {

TEST(DeepSeekMxfp4CodecTest, DecodesAllE2m1NibblesAtUnitScale) {
  constexpr std::array<double, 16> expected = {
      0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
  for (std::uint8_t bits = 0; bits < 16; ++bits) {
    auto decoded = DeepSeekMxfp4Codec::DecodeScalar(bits, 127);
    ASSERT_TRUE(decoded.ok());
    EXPECT_EQ(*decoded, expected[bits]);
    if (bits == 8) EXPECT_TRUE(std::signbit(*decoded));
  }
}

TEST(DeepSeekMxfp4CodecTest, ReinterpretsUe8m0BitsAsPowerOfTwo) {
  auto half = DeepSeekMxfp4Codec::DecodeScalar(2, 126);
  auto one = DeepSeekMxfp4Codec::DecodeScalar(2, 127);
  auto two = DeepSeekMxfp4Codec::DecodeScalar(2, 128);
  ASSERT_TRUE(half.ok()); ASSERT_TRUE(one.ok()); ASSERT_TRUE(two.ok());
  EXPECT_EQ(*half, 0.5);
  EXPECT_EQ(*one, 1.0);
  EXPECT_EQ(*two, 2.0);
  EXPECT_FALSE(DeepSeekMxfp4Codec::DecodeScalar(2, 0xFF).ok());
}

TEST(DeepSeekMxfp4CodecTest, UsesLowNibbleForLowerKIndex) {
  const std::array<std::byte, 2> packed = {
      std::byte{0x21}, std::byte{0xC7}};
  const std::array<std::byte, 1> scales = {std::byte{127}};
  auto decoded = DeepSeekMxfp4Codec::DecodeRow(packed, scales, 4);
  ASSERT_TRUE(decoded.ok());
  EXPECT_EQ(*decoded,
            (std::vector<float>{0.5F, 1.0F, 6.0F, -2.0F}));
}

TEST(DeepSeekMxfp4CodecTest, AppliesOneScalePerThirtyTwoKValues) {
  std::vector<std::byte> packed(17, std::byte{0x22});
  const std::array<std::byte, 2> scales = {
      std::byte{127}, std::byte{128}};
  auto decoded = DeepSeekMxfp4Codec::DecodeRow(packed, scales, 33);
  ASSERT_TRUE(decoded.ok());
  EXPECT_EQ((*decoded)[31], 1.0F);
  EXPECT_EQ((*decoded)[32], 2.0F);
}

TEST(DeepSeekMxfp4CodecTest, RejectsInvalidShapeScaleAndFloatOverflow) {
  const std::array<std::byte, 1> packed = {std::byte{0x22}};
  const std::array<std::byte, 1> valid = {std::byte{127}};
  const std::array<std::byte, 1> invalid = {std::byte{0xFF}};
  EXPECT_FALSE(DeepSeekMxfp4Codec::DecodeRow(packed, valid, 3).ok());
  EXPECT_FALSE(DeepSeekMxfp4Codec::DecodeRow(packed, invalid, 2).ok());
  const std::array<std::byte, 1> overflow = {std::byte{254}};
  const std::array<std::byte, 1> maximum = {std::byte{0x77}};
  EXPECT_FALSE(DeepSeekMxfp4Codec::DecodeRow(maximum, overflow, 2).ok());
}

}  // namespace
}  // namespace pih
