#include "pih/model/qwen3_int4_canonical_tensor.h"

#include <algorithm>
#include <cmath>

#include "pih/core/checked_math.h"
#include "pih/core/float16.h"

namespace pih {

std::int8_t QwenInt4CanonicalTensor::value(
    std::uint64_t row, std::uint64_t column) const noexcept {
  const std::uint64_t packed_columns = (columns_ + 1) / 2;
  const std::uint8_t byte = std::to_integer<std::uint8_t>(
      packed_values_[row * packed_columns + column / 2]);
  const std::uint8_t nibble =
      (column & 1U) == 0 ? byte & 0x0fU : byte >> 4U;
  return static_cast<std::int8_t>(nibble < 8 ? nibble : nibble - 16);
}

Result<QwenInt4CanonicalTensor> QwenInt4CanonicalTensor::Verify(
    std::span<const std::byte> packed_values,
    std::span<const std::uint16_t> scale_bits, std::uint64_t rows,
    std::uint64_t columns) {
  auto elements = checked_mul_u64(rows, columns);
  if (!elements.ok()) return elements.status();
  if (rows == 0 || columns == 0 || *elements > kMaximumElements) {
    return Status::InvalidArgument("Qwen INT4 canonical shape is invalid");
  }
  auto group_numerator = checked_add_u64(columns, kGroupSize - 1);
  if (!group_numerator.ok()) return group_numerator.status();
  const std::uint64_t groups = *group_numerator / kGroupSize;
  auto expected_scales = checked_mul_u64(rows, groups);
  if (!expected_scales.ok()) return expected_scales.status();
  auto packed_numerator = checked_add_u64(columns, 1);
  if (!packed_numerator.ok()) return packed_numerator.status();
  const std::uint64_t packed_columns = *packed_numerator / 2;
  auto expected_packed = checked_mul_u64(rows, packed_columns);
  if (!expected_packed.ok()) return expected_packed.status();
  if (packed_values.size() != *expected_packed ||
      scale_bits.size() != *expected_scales) {
    return Status::InvalidArgument(
        "Qwen INT4 canonical payload geometry is invalid");
  }
  if ((columns & 1U) != 0) {
    for (std::uint64_t row = 0; row < rows; ++row) {
      const auto tail = std::to_integer<std::uint8_t>(
          packed_values[row * packed_columns + packed_columns - 1]);
      if ((tail & 0xf0U) != 0) {
        return Status::InvalidArgument(
            "Qwen INT4 canonical padding nibble is nonzero");
      }
    }
  }
  QwenInt4CanonicalTensor result(
      rows, columns, groups,
      std::vector<std::byte>(packed_values.begin(), packed_values.end()),
      std::vector<std::uint16_t>(scale_bits.begin(), scale_bits.end()));
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t group = 0; group < groups; ++group) {
      const auto scale = Float16{scale_bits[row * groups + group]};
      const float numeric_scale = scale.to_float();
      if (!std::isfinite(numeric_scale) || numeric_scale <= 0.0F) {
        return Status::InvalidArgument(
            "Qwen INT4 canonical scale is not finite and positive");
      }
      const std::uint64_t first = group * kGroupSize;
      const std::uint64_t last = std::min(columns, first + kGroupSize);
      bool nonzero = false;
      for (std::uint64_t column = first; column < last; ++column) {
        const auto quantized = result.value(row, column);
        if (quantized == -8) {
          return Status::InvalidArgument(
              "Qwen INT4 canonical payload contains reserved -8");
        }
        nonzero = nonzero || quantized != 0;
      }
      if (!nonzero && scale.bits != 0x3c00U) {
        return Status::InvalidArgument(
            "Qwen INT4 zero group does not use canonical scale one");
      }
    }
  }
  return result;
}

Result<std::vector<float>> QwenInt4CanonicalTensor::dequantize() const {
  auto elements = checked_mul_u64(rows_, columns_);
  if (!elements.ok()) return elements.status();
  std::vector<float> output(static_cast<std::size_t>(*elements));
  for (std::uint64_t row = 0; row < rows_; ++row) {
    for (std::uint64_t column = 0; column < columns_; ++column) {
      const std::uint64_t group = column / kGroupSize;
      const float scale =
          Float16{scale_bits_[row * groups_per_row_ + group]}.to_float();
      output[row * columns_ + column] =
          scale * static_cast<float>(value(row, column));
    }
  }
  return output;
}

}  // namespace pih
