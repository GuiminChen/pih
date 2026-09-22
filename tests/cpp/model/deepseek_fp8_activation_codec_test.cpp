#include "pih/model/deepseek_fp8_activation_codec.h"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace pih {
namespace {

TEST(DeepSeekFp8ActivationCodecTest, EncodesE4m3FnCanonicalValues) {
  EXPECT_EQ(DeepSeekFp8ActivationCodec::EncodeE4m3Fn(0.0F).value(), 0x00);
  EXPECT_EQ(DeepSeekFp8ActivationCodec::EncodeE4m3Fn(-0.0F).value(), 0x80);
  EXPECT_EQ(DeepSeekFp8ActivationCodec::EncodeE4m3Fn(1.0F).value(), 0x38);
  EXPECT_EQ(DeepSeekFp8ActivationCodec::EncodeE4m3Fn(448.0F).value(), 0x7E);
  EXPECT_FLOAT_EQ(DeepSeekFp8ActivationCodec::DecodeE4m3Fn(0x7E).value(),
                  448.0F);
  EXPECT_FALSE(DeepSeekFp8ActivationCodec::DecodeE4m3Fn(0x7F).ok());
}

TEST(DeepSeekFp8ActivationCodecTest, UsesRoundToNearestTiesToEven) {
  // Halfway between 1.0 (even mantissa) and 1.125 (odd mantissa).
  EXPECT_EQ(DeepSeekFp8ActivationCodec::EncodeE4m3Fn(1.0625F).value(), 0x38);
  // Halfway between 1.125 (odd) and 1.25 (even).
  EXPECT_EQ(DeepSeekFp8ActivationCodec::EncodeE4m3Fn(1.1875F).value(), 0x3A);
}

TEST(DeepSeekFp8ActivationCodecTest, QuantizesEachTokenK128Block) {
  std::vector<float> input(256, 0.0F);
  input[0] = 448.0F;
  input[128] = 1.0F;
  auto quantized = DeepSeekFp8ActivationCodec::Quantize(input, 1, 256);
  ASSERT_TRUE(quantized.ok());
  ASSERT_EQ(quantized->scale_bits.size(), 2U);
  EXPECT_EQ(quantized->scale_bits[0], 127U);  // scale 1
  EXPECT_EQ(quantized->scale_bits[1], 119U);  // scale 2^-8
  EXPECT_EQ(quantized->e4m3_bits[0], 0x7E);
  EXPECT_EQ(quantized->e4m3_bits[128], 0x78);  // 1 / 2^-8 = 256
}

TEST(DeepSeekFp8ActivationCodecTest, AppliesOfficialZeroFloor) {
  std::vector<float> input(128, 0.0F);
  auto quantized = DeepSeekFp8ActivationCodec::Quantize(input, 1, 128);
  ASSERT_TRUE(quantized.ok());
  EXPECT_EQ(quantized->scale_bits, std::vector<std::uint8_t>({105U}));
}

TEST(DeepSeekFp8ActivationCodecTest, RejectsShapeAndNonfiniteInput) {
  std::vector<float> input(128, 0.0F);
  EXPECT_FALSE(DeepSeekFp8ActivationCodec::Quantize(input, 0, 128).ok());
  EXPECT_FALSE(DeepSeekFp8ActivationCodec::Quantize(input, 1, 64).ok());
  input.back() = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(DeepSeekFp8ActivationCodec::Quantize(input, 1, 128).ok());
}

}  // namespace
}  // namespace pih
