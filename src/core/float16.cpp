#include "pih/core/float16.h"

#include <bit>
#include <cmath>

namespace pih {

Float16 Float16::FromFloat(float value) noexcept {
  const std::uint32_t raw = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t sign = static_cast<std::uint16_t>((raw >> 16U) & 0x8000U);
  const std::uint32_t exponent = (raw >> 23U) & 0xffU;
  const std::uint32_t mantissa = raw & 0x7fffffU;
  if (exponent == 0xffU) {
    return Float16{static_cast<std::uint16_t>(
        sign | (mantissa == 0 ? 0x7c00U : 0x7e00U))};
  }
  const int unbiased = static_cast<int>(exponent) - 127;
  if (unbiased > 15) return Float16{static_cast<std::uint16_t>(sign | 0x7c00U)};
  if (unbiased >= -14) {
    std::uint32_t half_exponent = static_cast<std::uint32_t>(unbiased + 15);
    std::uint32_t half_mantissa = mantissa >> 13U;
    const std::uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U ||
        (remainder == 0x1000U && (half_mantissa & 1U) != 0)) {
      ++half_mantissa;
      if (half_mantissa == 0x400U) {
        half_mantissa = 0;
        ++half_exponent;
        if (half_exponent >= 31U) {
          return Float16{static_cast<std::uint16_t>(sign | 0x7c00U)};
        }
      }
    }
    return Float16{static_cast<std::uint16_t>(
        sign | (half_exponent << 10U) | half_mantissa)};
  }
  if (unbiased < -25) return Float16{sign};
  const std::uint32_t significant = 0x800000U | mantissa;
  const std::uint32_t shift = static_cast<std::uint32_t>(-14 - unbiased + 13);
  std::uint32_t half_mantissa = significant >> shift;
  const std::uint32_t mask = (UINT32_C(1) << shift) - 1U;
  const std::uint32_t remainder = significant & mask;
  const std::uint32_t halfway = UINT32_C(1) << (shift - 1U);
  if (remainder > halfway ||
      (remainder == halfway && (half_mantissa & 1U) != 0)) {
    ++half_mantissa;
  }
  return Float16{
      static_cast<std::uint16_t>(sign | half_mantissa)};
}

float Float16::to_float() const noexcept {
  const bool negative = (bits & 0x8000U) != 0;
  const std::uint16_t exponent = (bits >> 10U) & 0x1fU;
  const std::uint16_t mantissa = bits & 0x03ffU;
  float value = 0.0F;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 0x1fU) {
    value = mantissa == 0 ? INFINITY : NAN;
  } else {
    value = std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F,
                       static_cast<int>(exponent) - 15);
  }
  return negative ? -value : value;
}

}  // namespace pih
