#include "pih/model/deepseek_fp8_gemm_oracle.h"

#include <cmath>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<std::vector<float>> DeepSeekFp8GemmOracle::Multiply(
    const DeepSeekFp8Activation& activation,
    std::span<const std::byte> weight_e4m3,
    std::span<const std::byte> weight_scale_bits,
    std::uint32_t output_size) {
  const auto m = activation.token_count;
  const auto k = activation.logical_k;
  if (m == 0 || output_size == 0 || output_size % 128U != 0 ||
      k == 0 || k % 128U != 0) {
    return Status::InvalidArgument("DeepSeek FP8 GEMM geometry is invalid");
  }
  auto activation_elements = checked_mul_u64(m, k);
  auto weight_elements = checked_mul_u64(output_size, k);
  auto output_elements = checked_mul_u64(m, output_size);
  const auto k_blocks = k / 128U;
  auto activation_scales = checked_mul_u64(m, k_blocks);
  auto weight_scales = checked_mul_u64(output_size / 128U, k_blocks);
  if (!activation_elements.ok() || !weight_elements.ok() ||
      !output_elements.ok() || !activation_scales.ok() ||
      !weight_scales.ok() ||
      activation.e4m3_bits.size() != *activation_elements ||
      activation.scale_bits.size() != *activation_scales ||
      weight_e4m3.size() != *weight_elements ||
      weight_scale_bits.size() != *weight_scales ||
      *output_elements > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("DeepSeek FP8 GEMM shape is invalid");
  }
  std::vector<float> output(static_cast<std::size_t>(*output_elements));
  for (std::uint32_t row_m = 0; row_m < m; ++row_m) {
    for (std::uint32_t row_n = 0; row_n < output_size; ++row_n) {
      float accumulator = 0.0F;
      for (std::uint32_t block = 0; block < k_blocks; ++block) {
        const auto activation_scale_bits =
            activation.scale_bits[static_cast<std::size_t>(row_m) * k_blocks +
                                  block];
        const auto weight_scale = std::to_integer<std::uint8_t>(
            weight_scale_bits[static_cast<std::size_t>(row_n / 128U) *
                                  k_blocks +
                              block]);
        if (activation_scale_bits == 0xFFU || weight_scale == 0xFFU) {
          return Status::InvalidArgument(
              "DeepSeek FP8 GEMM UE8M0 scale is invalid");
        }
        float local = 0.0F;
        for (std::uint32_t offset = 0; offset < 128U; ++offset) {
          const auto column = block * 128U + offset;
          auto a = DeepSeekFp8ActivationCodec::DecodeE4m3Fn(
              activation.e4m3_bits[static_cast<std::size_t>(row_m) * k +
                                   column]);
          auto b = DeepSeekFp8ActivationCodec::DecodeE4m3Fn(
              std::to_integer<std::uint8_t>(
                  weight_e4m3[static_cast<std::size_t>(row_n) * k + column]));
          if (!a.ok()) return a.status();
          if (!b.ok()) return b.status();
          local += *a * *b;
        }
        accumulator += local *
                       std::ldexp(1.0F,
                                  static_cast<int>(activation_scale_bits) - 127) *
                       std::ldexp(1.0F, static_cast<int>(weight_scale) - 127);
        if (!std::isfinite(accumulator)) {
          return Status::InvalidArgument(
              "DeepSeek FP8 GEMM accumulation is nonfinite");
        }
      }
      output[static_cast<std::size_t>(row_m) * output_size + row_n] =
          accumulator;
    }
  }
  return output;
}

}  // namespace pih
