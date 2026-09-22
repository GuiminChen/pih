#pragma once

#include <bit>
#include <cstdint>

namespace pih {

// Storage type only: arithmetic is deliberately explicit through float so the
// CPU oracle and CUDA kernels share one visible rounding boundary.
struct BFloat16 final {
  std::uint16_t bits = 0;

  static BFloat16 FromFloat(float value) noexcept {
    const std::uint32_t raw = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t magnitude = raw & 0x7fffffffU;
    if (magnitude > 0x7f800000U) {
      // Never round a NaN payload into infinity. Canonicalize it to a quiet NaN
      // while retaining the sign bit.
      return BFloat16{static_cast<std::uint16_t>((raw >> 16U) | 0x0040U)};
    }
    const std::uint32_t rounded = raw + 0x7fffU + ((raw >> 16U) & 1U);
    return BFloat16{static_cast<std::uint16_t>(rounded >> 16U)};
  }

  [[nodiscard]] float to_float() const noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
  }

  friend bool operator==(const BFloat16&, const BFloat16&) = default;
};

static_assert(sizeof(BFloat16) == 2);

}  // namespace pih
