#include "pih/model/deepseek_fp4_gemm_oracle.h"

#include <cmath>
#include <limits>

#include "pih/core/checked_math.h"
#include "pih/model/deepseek_mxfp4_codec.h"

namespace pih {

Result<std::vector<float>> DeepSeekFp4GemmOracle::Multiply(
    const DeepSeekFp8Activation& activation,
    std::span<const std::byte> packed_weights,
    std::span<const std::byte> weight_scale_bits,
    std::uint32_t output_size) {
  const auto tokens = activation.token_count;
  const auto logical_k = activation.logical_k;
  if (tokens == 0 || output_size == 0 || logical_k == 0 ||
      logical_k % DeepSeekFp8ActivationCodec::kValuesPerScale != 0 ||
      logical_k % DeepSeekMxfp4Codec::kValuesPerScale != 0) {
    return Status::InvalidArgument("DeepSeek FP4 GEMM geometry is invalid");
  }
  auto activation_elements = checked_mul_u64(tokens, logical_k);
  auto packed_elements = checked_mul_u64(output_size, logical_k / 2U);
  auto weight_scales = checked_mul_u64(
      output_size, logical_k / DeepSeekMxfp4Codec::kValuesPerScale);
  auto activation_scales = checked_mul_u64(
      tokens, logical_k / DeepSeekFp8ActivationCodec::kValuesPerScale);
  auto output_elements = checked_mul_u64(tokens, output_size);
  if (!activation_elements.ok() || !packed_elements.ok() ||
      !weight_scales.ok() || !activation_scales.ok() ||
      !output_elements.ok() ||
      activation.e4m3_bits.size() != *activation_elements ||
      activation.scale_bits.size() != *activation_scales ||
      packed_weights.size() != *packed_elements ||
      weight_scale_bits.size() != *weight_scales ||
      *output_elements > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("DeepSeek FP4 GEMM shape is invalid");
  }

  std::vector<float> output(static_cast<std::size_t>(*output_elements));
  const auto k32_count = logical_k / DeepSeekMxfp4Codec::kValuesPerScale;
  const auto k128_count =
      logical_k / DeepSeekFp8ActivationCodec::kValuesPerScale;
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint32_t row = 0; row < output_size; ++row) {
      float accumulator = 0.0F;
      for (std::uint32_t block = 0; block < k32_count; ++block) {
        const auto activation_scale_bits =
            activation.scale_bits[static_cast<std::size_t>(token) *
                                      k128_count +
                                  block / 4U];
        const auto weight_scale_byte = std::to_integer<std::uint8_t>(
            weight_scale_bits[static_cast<std::size_t>(row) * k32_count +
                              block]);
        if (activation_scale_bits == 0xFFU || weight_scale_byte == 0xFFU) {
          return Status::InvalidArgument(
              "DeepSeek FP4 GEMM UE8M0 scale is invalid");
        }
        const auto activation_scale =
            std::ldexp(1.0F, static_cast<int>(activation_scale_bits) - 127);
        const auto weight_scale =
            std::ldexp(1.0F, static_cast<int>(weight_scale_byte) - 127);
        float local = 0.0F;
        for (std::uint32_t offset = 0;
             offset < DeepSeekMxfp4Codec::kValuesPerScale; ++offset) {
          const auto k = block * DeepSeekMxfp4Codec::kValuesPerScale + offset;
          auto a = DeepSeekFp8ActivationCodec::DecodeE4m3Fn(
              activation.e4m3_bits[static_cast<std::size_t>(token) * logical_k +
                                   k]);
          const auto packed = std::to_integer<std::uint8_t>(
              packed_weights[static_cast<std::size_t>(row) * (logical_k / 2U) +
                             k / 2U]);
          const auto nibble = static_cast<std::uint8_t>(
              (k & 1U) == 0 ? packed & 0x0FU : packed >> 4U);
          auto b = DeepSeekMxfp4Codec::DecodeScalar(nibble, 127U);
          if (!a.ok()) return a.status();
          if (!b.ok()) return b.status();
          local += *a * static_cast<float>(*b);
        }
        accumulator += local * activation_scale * weight_scale;
        if (!std::isfinite(accumulator)) {
          return Status::InvalidArgument(
              "DeepSeek FP4 GEMM accumulation is nonfinite");
        }
      }
      output[static_cast<std::size_t>(token) * output_size + row] = accumulator;
    }
  }
  return output;
}

}  // namespace pih
