#include "pih/model/deepseek_mxfp4_codec.h"

#include <array>
#include <cmath>
#include <limits>

namespace pih {

Result<double> DeepSeekMxfp4Codec::DecodeScalar(std::uint8_t e2m1_bits,
                                                std::uint8_t ue8m0_bits) {
  if (e2m1_bits > 0x0F || ue8m0_bits == kInvalidScaleBits) {
    return Status::InvalidArgument("DeepSeek MXFP4 encoded bits are invalid");
  }
  static constexpr std::array<double, 8> kMagnitude = {
      0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  const auto magnitude = kMagnitude[e2m1_bits & 0x07U];
  const auto signed_value = (e2m1_bits & 0x08U) != 0 ? -magnitude : magnitude;
  return std::ldexp(signed_value, static_cast<int>(ue8m0_bits) - 127);
}

Result<std::vector<float>> DeepSeekMxfp4Codec::DecodeRow(
    std::span<const std::byte> packed,
    std::span<const std::byte> scale_bits, std::uint32_t logical_k) {
  if (logical_k == 0 || packed.size() != (logical_k + 1U) / 2U ||
      scale_bits.size() != (logical_k + kValuesPerScale - 1U) /
                               kValuesPerScale) {
    return Status::InvalidArgument("DeepSeek MXFP4 row shape is invalid");
  }
  std::vector<float> result(logical_k);
  for (std::uint32_t k = 0; k < logical_k; ++k) {
    const auto byte = std::to_integer<std::uint8_t>(packed[k / 2U]);
    const auto nibble = static_cast<std::uint8_t>(
        (k & 1U) == 0 ? byte & 0x0FU : byte >> 4U);
    const auto scale =
        std::to_integer<std::uint8_t>(scale_bits[k / kValuesPerScale]);
    auto decoded = DecodeScalar(nibble, scale);
    if (!decoded.ok()) return decoded.status();
    if (!std::isfinite(*decoded) ||
        std::abs(*decoded) > std::numeric_limits<float>::max()) {
      return Status::InvalidArgument(
          "DeepSeek MXFP4 value overflows compatibility float");
    }
    result[k] = static_cast<float>(*decoded);
  }
  return result;
}

}  // namespace pih
