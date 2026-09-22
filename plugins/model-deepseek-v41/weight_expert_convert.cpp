#include "weight_expert_convert.h"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <new>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "V4.1 offline expert conversion requires strict floating-point semantics"
#endif

namespace pih::deepseek_v41 {
namespace {
double PositiveFp8(unsigned code) {
  const unsigned exponent = code >> 3, mantissa = code & 7;
  return exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -9) :
      std::ldexp(static_cast<double>(8 + mantissa), static_cast<int>(exponent) - 10);
}
unsigned RoundFp8(double value) {
  // All caller values are nonnegative and <=384, strictly below max finite 448.
  unsigned low = 0, high = 126;
  while (low < high) {
    const unsigned middle = (low + high) / 2;
    if (PositiveFp8(middle) < value) low = middle + 1;
    else high = middle;
  }
  if (low == 0) return 0;
  const double midpoint = (PositiveFp8(low - 1) + PositiveFp8(low)) / 2;
  if (value < midpoint) return low - 1;
  if (value > midpoint) return low;
  return (low & 1U) == 0 ? low : low - 1;
}
}
Result<ExpertFp8Block> ConvertExpertFp4Block(
    std::span<const std::byte> packed, std::span<const std::byte> scales) {
  if (packed.size() != 32 * 16 || scales.size() != 32)
    return Status::InvalidArgument("V4.1 expert FP4 block requires 512 packed bytes and 32 E8M0 scales");
  if (std::fegetround() != FE_TONEAREST)
    return Status::FailedPrecondition("V4.1 expert conversion requires round-to-nearest mode");
  unsigned maximum = 0;
  for (const auto scale : scales) {
    const auto code = std::to_integer<unsigned>(scale);
    if (code == 255) return Status::InvalidArgument("V4.1 expert source scale is NaN");
    maximum = std::max(maximum, code);
  }
  if (maximum < 6)
    return Status::InvalidArgument("V4.1 expert shared scale max/64 is below E8M0 range");
  ExpertFp8Block result;
  result.scale = static_cast<std::byte>(maximum - 6);
  constexpr std::array<double, 8> magnitudes{0, 0.5, 1, 1.5, 2, 3, 4, 6};
  for (unsigned row = 0; row < 32; ++row) {
    const int shift = static_cast<int>(std::to_integer<unsigned>(scales[row])) - static_cast<int>(maximum) + 6;
    for (unsigned column = 0; column < 32; ++column) {
      const auto byte = std::to_integer<unsigned>(packed[row * 16 + column / 2]);
      const auto nibble = (byte >> ((column & 1U) * 4)) & 15U;
      const auto magnitude = nibble & 7U;
      const double value = std::ldexp(magnitudes[magnitude], shift);
      auto code = RoundFp8(value);
      // The reference table maps both FP4 zero codes to positive zero. Nonzero
      // negative inputs retain the sign even if FP8 rounding produces zero.
      if ((nibble & 8U) && magnitude != 0) code |= 128U;
      result.weight[row * 32 + column] = static_cast<std::byte>(code);
    }
  }
  return result;
}
Status ConvertExpertFp4Tensor(std::uint64_t rows, std::uint64_t columns,
    const WoATensorReader& read_weight, const WoATensorReader& read_scales,
    const WoATensorWriter& write_weight, const WoATensorWriter& write_scales) {
  if (!((rows == 2304 && columns == 5120) || (rows == 5120 && columns == 2304)) ||
      !read_weight || !read_scales || !write_weight || !write_scales)
    return Status::InvalidArgument("V4.1 expert conversion matrix shape/callback invalid");
  try {
    const auto packed_columns = columns / 2, scale_columns = columns / 32;
    std::vector<std::byte> input(32 * packed_columns), scales(32 * scale_columns);
    std::vector<std::byte> output(32 * columns), output_scales(scale_columns);
    std::array<std::byte, 512> block;
    std::array<std::byte, 32> block_scales;
    for (std::uint64_t first = 0; first < rows; first += 32) {
      auto status = read_weight(first * packed_columns, input); if (!status.ok()) return status;
      status = read_scales(first * scale_columns, scales); if (!status.ok()) return status;
      for (std::uint64_t group = 0; group < scale_columns; ++group) {
        for (std::uint64_t row = 0; row < 32; ++row) {
          std::copy_n(input.begin() + row * packed_columns + group * 16, 16, block.begin() + row * 16);
          block_scales[row] = scales[row * scale_columns + group];
        }
        auto converted = ConvertExpertFp4Block(block, block_scales);
        if (!converted.ok()) return converted.status();
        for (std::uint64_t row = 0; row < 32; ++row)
          std::copy_n(converted->weight.begin() + row * 32, 32, output.begin() + row * columns + group * 32);
        output_scales[group] = converted->scale;
      }
      status = write_weight(first * columns, output); if (!status.ok()) return status;
      status = write_scales((first / 32) * scale_columns, output_scales); if (!status.ok()) return status;
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 expert conversion allocation failed; discard both unpublished outputs");
  } catch (...) {
    return Status::Internal("V4.1 expert conversion callback failed; discard both unpublished outputs");
  }
}
}  // namespace pih::deepseek_v41
