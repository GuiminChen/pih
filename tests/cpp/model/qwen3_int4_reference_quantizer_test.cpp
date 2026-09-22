#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "../../../plugins/offline-qwen/qwen3_int4_reference_quantizer.h"

namespace pih {
namespace {

BFloat16 bf(float value) { return BFloat16::FromFloat(value); }

TEST(QwenInt4ReferenceQuantizerTest, FreezesZeroGroupScaleAndPadding) {
  const std::array<BFloat16, 3> source{bf(0.0F), bf(-0.0F), bf(0.0F)};
  auto result = QwenInt4QuantizedTensor::Quantize(source, 1, 3);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->groups_per_row(), 1);
  EXPECT_EQ(result->scale_bits()[0], 0x3c00);
  EXPECT_EQ(std::vector(result->logical_values().begin(),
                        result->logical_values().end()),
            (std::vector<std::int8_t>{0, 0, 0}));
  EXPECT_EQ(std::vector(result->packed_values().begin(),
                        result->packed_values().end()),
            (std::vector<std::byte>{std::byte{0}, std::byte{0}}));
}

TEST(QwenInt4ReferenceQuantizerTest, PacksSignedNibblesInCanonicalKOrder) {
  const std::array source{bf(-7), bf(-6), bf(-1), bf(0),
                          bf(1),  bf(6),  bf(7)};
  auto result = QwenInt4QuantizedTensor::Quantize(source, 1, source.size());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->scale_bits()[0], 0x3c00);
  EXPECT_EQ(std::vector(result->packed_values().begin(),
                        result->packed_values().end()),
            (std::vector<std::byte>{std::byte{0xa9}, std::byte{0x0f},
                                    std::byte{0x61}, std::byte{0x07}}));
}

TEST(QwenInt4ReferenceQuantizerTest, RoundsPositiveAndNegativeTiesToEven) {
  const std::array source{bf(0.5F),  bf(1.5F),  bf(2.5F),  bf(3.5F), bf(7.0F),
                          bf(-0.5F), bf(-1.5F), bf(-2.5F), bf(-3.5F)};
  auto result = QwenInt4QuantizedTensor::Quantize(source, 1, source.size());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(std::vector(result->logical_values().begin(),
                        result->logical_values().end()),
            (std::vector<std::int8_t>{0, 2, 2, 4, 7, 0, -2, -2, -4}));
}

TEST(QwenInt4ReferenceQuantizerTest, RejectsNonfiniteAndScaleUnderflow) {
  const std::array nonfinite{BFloat16{0x7f80}};
  EXPECT_FALSE(QwenInt4QuantizedTensor::Quantize(nonfinite, 1, 1).ok());
  const std::array tiny{BFloat16{0x0001}};
  EXPECT_FALSE(QwenInt4QuantizedTensor::Quantize(tiny, 1, 1).ok());
}

TEST(QwenInt4ReferenceQuantizerTest, RejectsMalformedOrUnboundedShapes) {
  const std::vector<BFloat16> one{bf(1)};
  EXPECT_FALSE(QwenInt4QuantizedTensor::Quantize(one, 0, 1).ok());
  EXPECT_FALSE(QwenInt4QuantizedTensor::Quantize(one, 1, 2).ok());
  EXPECT_FALSE(QwenInt4QuantizedTensor::Quantize(
                   {}, 1, QwenInt4QuantizedTensor::kMaximumElements + 1)
                   .ok());
}

}  // namespace
}  // namespace pih
