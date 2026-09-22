#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_int4_canonical_tensor.h"
#include "../../../plugins/offline-qwen/qwen3_int4_reference_quantizer.h"

namespace pih {
namespace {

BFloat16 bf(float value) { return BFloat16::FromFloat(value); }

TEST(QwenInt4CanonicalTensorTest, IndependentlyVerifiesAndDequantizesReference) {
  const std::array source{bf(-7), bf(-1), bf(0), bf(3), bf(7)};
  const auto quantized =
      QwenInt4QuantizedTensor::Quantize(source, 1, source.size()).value();
  auto verified = QwenInt4CanonicalTensor::Verify(
      quantized.packed_values(), quantized.scale_bits(), 1, source.size());
  ASSERT_TRUE(verified.ok()) << verified.status().message();
  const auto values = verified->dequantize().value();
  EXPECT_EQ(values, (std::vector<float>{-7, -1, 0, 3, 7}));
  EXPECT_EQ(verified->groups_per_row(), 1);
}

TEST(QwenInt4CanonicalTensorTest, RejectsReservedNibbleAndTailPadding) {
  const std::array scales{std::uint16_t{0x3c00}};
  const std::array reserved{std::byte{0x08}};
  EXPECT_FALSE(QwenInt4CanonicalTensor::Verify(reserved, scales, 1, 1).ok());
  const std::array bad_tail{std::byte{0x10}};
  EXPECT_FALSE(QwenInt4CanonicalTensor::Verify(bad_tail, scales, 1, 1).ok());
}

TEST(QwenInt4CanonicalTensorTest, RejectsInvalidScaleAndNoncanonicalZeroGroup) {
  const std::array zero{std::byte{0x00}};
  const std::array zero_scale{std::uint16_t{0x0000}};
  EXPECT_FALSE(QwenInt4CanonicalTensor::Verify(zero, zero_scale, 1, 2).ok());
  const std::array infinite_scale{std::uint16_t{0x7c00}};
  EXPECT_FALSE(
      QwenInt4CanonicalTensor::Verify(zero, infinite_scale, 1, 2).ok());
  const std::array noncanonical_zero{std::uint16_t{0x4000}};
  EXPECT_FALSE(
      QwenInt4CanonicalTensor::Verify(zero, noncanonical_zero, 1, 2).ok());
}

TEST(QwenInt4CanonicalTensorTest, RejectsPayloadGeometryMismatch) {
  const std::array packed{std::byte{0}};
  const std::array scales{std::uint16_t{0x3c00}};
  EXPECT_FALSE(QwenInt4CanonicalTensor::Verify({}, scales, 1, 2).ok());
  EXPECT_FALSE(QwenInt4CanonicalTensor::Verify(packed, {}, 1, 2).ok());
  EXPECT_FALSE(QwenInt4CanonicalTensor::Verify(packed, scales, 0, 2).ok());
}

}  // namespace
}  // namespace pih
