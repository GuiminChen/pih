// Offline plugin-owned conversion implementation.
#include "qwen3_int4_reference_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cfenv>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Canonical Qwen INT4 quantization requires finite checks and strict floating-point semantics"
#endif

#include "pih/core/checked_math.h"
#include "pih/core/float16.h"

namespace pih {
namespace {

int round_ties_to_even(float value) {
  const bool negative = std::signbit(value);
  const float magnitude = std::abs(value);
  const float base_float = std::floor(magnitude);
  int base = static_cast<int>(base_float);
  const float fraction = magnitude - base_float;
  if (fraction > 0.5F || (fraction == 0.5F && (base & 1) != 0)) ++base;
  return negative ? -base : base;
}

}  // namespace

Result<QwenInt4QuantizedTensor> QwenInt4QuantizedTensor::Quantize(
    std::span<const BFloat16> source, std::uint64_t rows,
    std::uint64_t columns) {
  if (std::fegetround() != FE_TONEAREST) {
    return Status::FailedPrecondition(
        "Qwen INT4 quantization requires round-to-nearest floating-point mode");
  }
  auto elements = checked_mul_u64(rows, columns);
  if (!elements.ok()) return elements.status();
  if (rows == 0 || columns == 0 || *elements > kMaximumElements ||
      source.size() != *elements) {
    return Status::InvalidArgument(
        "Qwen INT4 source tensor shape is invalid");
  }
  auto group_numerator = checked_add_u64(columns, kGroupSize - 1);
  if (!group_numerator.ok()) return group_numerator.status();
  const std::uint64_t groups = *group_numerator / kGroupSize;
  auto scale_count = checked_mul_u64(rows, groups);
  if (!scale_count.ok()) return scale_count.status();
  auto packed_columns_numerator = checked_add_u64(columns, 1);
  if (!packed_columns_numerator.ok()) return packed_columns_numerator.status();
  const std::uint64_t packed_columns = *packed_columns_numerator / 2;
  auto packed_count = checked_mul_u64(rows, packed_columns);
  if (!packed_count.ok()) return packed_count.status();

  std::vector<std::int8_t> logical(static_cast<std::size_t>(*elements), 0);
  std::vector<std::uint16_t> scales(
      static_cast<std::size_t>(*scale_count), 0);
  std::vector<std::byte> packed(static_cast<std::size_t>(*packed_count),
                                std::byte{0});
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t group = 0; group < groups; ++group) {
      const std::uint64_t first = group * kGroupSize;
      const std::uint64_t last = std::min(columns, first + kGroupSize);
      float maximum = 0.0F;
      for (std::uint64_t column = first; column < last; ++column) {
        const float value = source[row * columns + column].to_float();
        if (!std::isfinite(value)) {
          return Status::InvalidArgument(
              "Qwen INT4 source tensor contains a nonfinite value");
        }
        maximum = std::max(maximum, std::abs(value));
      }
      Float16 scale{0x3c00U};
      if (maximum != 0.0F) {
        scale = Float16::FromFloat(maximum / 7.0F);
        const float restored = scale.to_float();
        if (!std::isfinite(restored) || restored == 0.0F) {
          return Status::InvalidArgument(
              "Qwen INT4 nonzero group scale is outside FP16 range");
        }
      }
      scales[row * groups + group] = scale.bits;
      const float divisor = scale.to_float();
      for (std::uint64_t column = first; column < last; ++column) {
        const float value = source[row * columns + column].to_float();
        int quantized = maximum == 0.0F
                            ? 0
                            : round_ties_to_even(value / divisor);
        quantized = std::clamp(quantized, -7, 7);
        logical[row * columns + column] =
            static_cast<std::int8_t>(quantized);
      }
    }
    for (std::uint64_t column = 0; column < columns; ++column) {
      const std::uint8_t nibble = static_cast<std::uint8_t>(
                                      logical[row * columns + column]) &
                                  0x0fU;
      if (nibble == 0x08U) {
        return Status::Internal("Qwen INT4 canonical pack emitted reserved -8");
      }
      const std::uint64_t byte_index = row * packed_columns + column / 2;
      const unsigned shift = (column & 1U) == 0 ? 0U : 4U;
      packed[byte_index] |= static_cast<std::byte>(nibble << shift);
    }
  }
  return QwenInt4QuantizedTensor(rows, columns, groups, std::move(logical),
                                 std::move(scales), std::move(packed));
}

}  // namespace pih
