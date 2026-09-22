#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

class DeepSeekMxfp4Codec final {
 public:
  static constexpr std::uint32_t kValuesPerScale = 32;
  static constexpr std::uint8_t kInvalidScaleBits = 0xFF;

  static Result<double> DecodeScalar(std::uint8_t e2m1_bits,
                                     std::uint8_t ue8m0_bits);
  static Result<std::vector<float>> DecodeRow(
      std::span<const std::byte> packed,
      std::span<const std::byte> scale_bits, std::uint32_t logical_k);
};

}  // namespace pih
