#include "weight_dequantize.h"
#include "pih/core/bfloat16.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <new>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "V4.1 offline weight conversion requires strict floating-point semantics"
#endif

namespace pih::deepseek_v41 {
Result<float> DecodeWeightE8M0(std::uint8_t bits) {
  if (bits == 255) return Status::InvalidArgument("V4.1 E8M0 scale is NaN");
  // Construct even the subnormal 2^-127 without floating arithmetic/FTZ.
  return std::bit_cast<float>(bits == 0 ? std::uint32_t{0x00400000} :
      static_cast<std::uint32_t>(bits) << 23);
}
Status DequantizeWoABlock(std::span<const std::byte> source, float scale,
    std::uint32_t block, std::span<std::byte> destination) {
  if ((block != 32 && block != 128) || source.size() != block * block ||
      destination.size() != 2 * source.size() || !std::isfinite(scale) || scale <= 0)
    return Status::InvalidArgument("V4.1 wo_a block geometry or scale invalid");
  if (std::fegetround() != FE_TONEAREST)
    return Status::FailedPrecondition("V4.1 wo_a conversion requires round-to-nearest floating-point mode");
#if defined(__x86_64__) || defined(_M_X64)
  if (_mm_getcsr() & ((1U << 15) | (1U << 6)))
    return Status::FailedPrecondition("V4.1 wo_a conversion requires gradual underflow (FTZ/DAZ disabled)");
#endif
  std::array<std::byte, 128 * 128 * 2> output;
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto bits = std::to_integer<unsigned>(source[i]);
    const auto magnitude = bits & 127U;
    if (magnitude == 127)
      return Status::InvalidArgument("V4.1 wo_a source contains E4M3FN NaN");
    const auto exponent = magnitude >> 3;
    const auto mantissa = magnitude & 7;
    float value = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9) :
        std::ldexp(static_cast<float>(8 + mantissa), static_cast<int>(exponent) - 10);
    if (bits & 128U) value = -value;
    const float scaled = value * scale;
    if (!std::isfinite(scaled))
      return Status::InvalidArgument("V4.1 wo_a FP32 dequantization overflow");
    const auto bf16 = BFloat16::FromFloat(scaled);
    if (!std::isfinite(bf16.to_float()))
      return Status::InvalidArgument("V4.1 wo_a BF16 dequantization overflow");
    output[2 * i] = static_cast<std::byte>(bf16.bits & 255U);
    output[2 * i + 1] = static_cast<std::byte>(bf16.bits >> 8);
  }
  std::copy_n(output.begin(), destination.size(), destination.begin());
  return Status::Ok();
}
Status DequantizeWoATensor(const WoASourceGeometry& g,
    const WoATensorReader& read_weight, const WoATensorReader& read_scale,
    const WoATensorWriter& write) {
  constexpr std::uint64_t rows = 8192, columns = 4096;
  if (!read_weight || !read_scale || !write || g.rows != rows || g.columns != columns ||
      g.weight_bytes != rows * columns || !g.scale_rows || !g.scale_columns ||
      rows % g.scale_rows || columns % g.scale_columns)
    return Status::InvalidArgument("V4.1 wo_a tensor source geometry or callbacks invalid");
  const auto block = rows / g.scale_rows;
  if ((block != 32 && block != 128) || columns / g.scale_columns != block)
    return Status::InvalidArgument("V4.1 wo_a requires square 32 or 128 scale blocks");
  std::uint64_t scale_element = 0;
  switch (g.scale_storage) {
    case WoAScaleStorage::kE8M0: scale_element = 1; break;
    case WoAScaleStorage::kF32LittleEndian: scale_element = 4; break;
    default: return Status::InvalidArgument("V4.1 wo_a scale storage unsupported");
  }
  if (g.scale_bytes != g.scale_rows * g.scale_columns * scale_element)
    return Status::InvalidArgument("V4.1 wo_a scale byte count differs from geometry");
  try {
    std::vector<std::byte> input_band(static_cast<std::size_t>(block * columns));
    std::vector<std::byte> output_band(input_band.size() * 2);
    std::vector<std::byte> input_block(static_cast<std::size_t>(block * block));
    std::vector<std::byte> output_block(input_block.size() * 2);
    std::vector<std::byte> scales(static_cast<std::size_t>(g.scale_columns * scale_element));
    for (std::uint64_t first_row = 0; first_row < rows; first_row += block) {
      auto status = read_weight(first_row * columns, input_band);
      if (!status.ok()) return status;
      status = read_scale((first_row / block) * scales.size(), scales);
      if (!status.ok()) return status;
      for (std::uint64_t first_column = 0; first_column < columns; first_column += block) {
        const auto scale_offset = (first_column / block) * scale_element;
        float scale = 0;
        if (g.scale_storage == WoAScaleStorage::kE8M0) {
          auto decoded = DecodeWeightE8M0(std::to_integer<std::uint8_t>(scales[scale_offset]));
          if (!decoded.ok()) return decoded.status();
          scale = *decoded;
        } else {
          std::uint32_t bits = 0;
          for (unsigned byte = 0; byte < 4; ++byte)
            bits |= std::to_integer<std::uint32_t>(scales[scale_offset + byte]) << (8 * byte);
          scale = std::bit_cast<float>(bits);
        }
        for (std::uint64_t row = 0; row < block; ++row)
          std::copy_n(input_band.begin() + row * columns + first_column, block,
              input_block.begin() + row * block);
        status = DequantizeWoABlock(input_block, scale, static_cast<std::uint32_t>(block), output_block);
        if (!status.ok()) return status;
        for (std::uint64_t row = 0; row < block; ++row)
          std::copy_n(output_block.begin() + 2 * row * block, 2 * block,
              output_band.begin() + 2 * (row * columns + first_column));
      }
      status = write(2 * first_row * columns, output_band);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 wo_a conversion allocation failed; discard unpublished output");
  } catch (...) {
    return Status::Internal("V4.1 wo_a conversion callback failed; discard unpublished output");
  }
}
}  // namespace pih::deepseek_v41
