#include "pih/model/deepseek_fp8_activation_codec.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<float> DeepSeekFp8ActivationCodec::DecodeE4m3Fn(std::uint8_t bits) {
  const auto exponent = static_cast<std::uint8_t>((bits >> 3U) & 0x0FU);
  const auto mantissa = static_cast<std::uint8_t>(bits & 0x07U);
  if (exponent == 0x0FU && mantissa == 0x07U) {
    return Status::InvalidArgument("DeepSeek E4M3FN NaN encoding is invalid");
  }
  float magnitude = 0.0F;
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<float>(mantissa), -9);
  } else {
    magnitude = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
  }
  return (bits & 0x80U) != 0 ? -magnitude : magnitude;
}

Result<std::uint8_t> DeepSeekFp8ActivationCodec::EncodeE4m3Fn(float value) {
  if (!std::isfinite(value) || std::abs(value) > kMaxFinite) {
    return Status::InvalidArgument("DeepSeek E4M3FN input is out of range");
  }
  if (value == 0.0F) return std::signbit(value) ? 0x80U : 0x00U;

  const auto sign = std::signbit(value) ? 0x80U : 0x00U;
  const auto magnitude = std::abs(value);
  std::uint8_t best = sign;
  double best_distance = std::numeric_limits<double>::infinity();
  for (std::uint16_t payload = 0; payload <= 0x7EU; ++payload) {
    if (payload == 0x7FU) continue;
    auto decoded = DecodeE4m3Fn(static_cast<std::uint8_t>(sign | payload));
    if (!decoded.ok()) continue;
    const auto distance =
        std::abs(static_cast<double>(magnitude) - std::abs(*decoded));
    if (distance < best_distance ||
        (distance == best_distance && (payload & 1U) == 0U &&
         (best & 1U) != 0U)) {
      best_distance = distance;
      best = static_cast<std::uint8_t>(sign | payload);
    }
  }
  return best;
}

Result<DeepSeekFp8Activation> DeepSeekFp8ActivationCodec::Quantize(
    std::span<const float> input, std::uint32_t token_count,
    std::uint32_t logical_k) {
  if (token_count == 0 || logical_k == 0 ||
      logical_k % kValuesPerScale != 0) {
    return Status::InvalidArgument(
        "DeepSeek FP8 activation geometry is invalid");
  }
  auto element_count = checked_mul_u64(token_count, logical_k);
  if (!element_count.ok() || *element_count != input.size()) {
    return Status::InvalidArgument("DeepSeek FP8 activation shape is invalid");
  }

  DeepSeekFp8Activation output;
  output.token_count = token_count;
  output.logical_k = logical_k;
  output.e4m3_bits.resize(input.size());
  output.scale_bits.resize(static_cast<std::size_t>(token_count) *
                           (logical_k / kValuesPerScale));
  for (std::uint32_t token = 0; token < token_count; ++token) {
    for (std::uint32_t group = 0; group < logical_k / kValuesPerScale;
         ++group) {
      const auto begin = static_cast<std::size_t>(token) * logical_k +
                         static_cast<std::size_t>(group) * kValuesPerScale;
      float amax = 0.0F;
      for (std::uint32_t offset = 0; offset < kValuesPerScale; ++offset) {
        const auto value = input[begin + offset];
        if (!std::isfinite(value)) {
          return Status::InvalidArgument(
              "DeepSeek FP8 activation input is nonfinite");
        }
        amax = std::max(amax, std::abs(value));
      }
      amax = std::max(amax, kAmaxFloor);
      const auto scale_exponent = static_cast<int>(
          std::ceil(std::log2(static_cast<double>(amax) / kMaxFinite)));
      if (scale_exponent < -127 || scale_exponent > 127) {
        return Status::InvalidArgument(
            "DeepSeek FP8 activation scale is not UE8M0 finite");
      }
      const auto scale = std::ldexp(1.0F, scale_exponent);
      output.scale_bits[static_cast<std::size_t>(token) *
                            (logical_k / kValuesPerScale) +
                        group] = static_cast<std::uint8_t>(scale_exponent + 127);
      for (std::uint32_t offset = 0; offset < kValuesPerScale; ++offset) {
        const auto normalized =
            std::clamp(input[begin + offset] / scale, -kMaxFinite, kMaxFinite);
        auto encoded = EncodeE4m3Fn(normalized);
        if (!encoded.ok()) return encoded.status();
        output.e4m3_bits[begin + offset] = *encoded;
      }
    }
  }
  return output;
}

}  // namespace pih
