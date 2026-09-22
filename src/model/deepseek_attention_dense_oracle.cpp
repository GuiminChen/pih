#include "pih/model/deepseek_attention_dense_oracle.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "pih/model/deepseek_fp8_activation_codec.h"

namespace pih {

Status deepseek_head_rms_oracle(
    std::span<const BFloat16> input, std::uint32_t tokens,
    std::uint32_t heads, std::uint32_t dimension, float epsilon,
    std::span<BFloat16> output) {
  const auto elements = static_cast<std::size_t>(tokens) * heads * dimension;
  if (tokens == 0 || heads == 0 || dimension == 0 || input.size() != elements ||
      output.size() != elements || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Status::InvalidArgument("DeepSeek head RMS oracle is invalid");
  }
  std::vector<BFloat16> result(elements);
  for (std::uint32_t token = 0; token < tokens; ++token)
    for (std::uint32_t head = 0; head < heads; ++head) {
      const auto offset = (static_cast<std::size_t>(token) * heads + head) * dimension;
      float squares = 0.0F;
      for (std::uint32_t column = 0; column < dimension; ++column) {
        const auto value = input[offset + column].to_float();
        if (!std::isfinite(value))
          return Status::InvalidArgument("DeepSeek head RMS input is non-finite");
        squares += value * value;
      }
      const auto inverse = 1.0F / std::sqrt(squares / dimension + epsilon);
      for (std::uint32_t column = 0; column < dimension; ++column)
        result[offset + column] = BFloat16::FromFloat(
            input[offset + column].to_float() * inverse);
    }
  std::copy(result.begin(), result.end(), output.begin());
  return Status::Ok();
}

Status deepseek_rotary_oracle(
    std::span<const BFloat16> input, std::span<const float> frequencies,
    std::uint32_t tokens, std::uint32_t heads, std::uint32_t dimension,
    std::uint32_t rope_dimension, bool inverse,
    std::span<BFloat16> output) {
  const auto elements = static_cast<std::size_t>(tokens) * heads * dimension;
  if (tokens == 0 || heads == 0 || dimension == 0 || rope_dimension == 0 ||
      rope_dimension > dimension || rope_dimension % 2 != 0 ||
      input.size() != elements || output.size() != elements ||
      frequencies.size() != static_cast<std::size_t>(tokens) * rope_dimension) {
    return Status::InvalidArgument("DeepSeek rotary oracle is invalid");
  }
  std::vector<BFloat16> result(input.begin(), input.end());
  const auto pairs = rope_dimension / 2;
  const auto rope_begin = dimension - rope_dimension;
  for (std::uint32_t token = 0; token < tokens; ++token)
    for (std::uint32_t head = 0; head < heads; ++head)
      for (std::uint32_t pair = 0; pair < pairs; ++pair) {
        const auto offset = (static_cast<std::size_t>(token) * heads + head) *
                            dimension + rope_begin + pair * 2U;
        const auto cosine = frequencies[static_cast<std::size_t>(token) *
                                            rope_dimension + pair];
        auto sine = frequencies[static_cast<std::size_t>(token) * rope_dimension +
                                pairs + pair];
        if (inverse) sine = -sine;
        const auto real = input[offset].to_float();
        const auto imag = input[offset + 1].to_float();
        if (!std::isfinite(real) || !std::isfinite(imag) ||
            !std::isfinite(cosine) || !std::isfinite(sine))
          return Status::InvalidArgument("DeepSeek rotary input is non-finite");
        result[offset] = BFloat16::FromFloat(real * cosine - imag * sine);
        result[offset + 1] = BFloat16::FromFloat(real * sine + imag * cosine);
      }
  std::copy(result.begin(), result.end(), output.begin());
  return Status::Ok();
}

Status deepseek_kv_fp8_simulate_oracle(
    std::span<const BFloat16> input, std::uint32_t tokens,
    std::uint32_t dimension, std::uint32_t quantized_dimension,
    std::uint32_t group_size, std::span<BFloat16> output) {
  const auto elements = static_cast<std::size_t>(tokens) * dimension;
  if (tokens == 0 || dimension == 0 || quantized_dimension == 0 ||
      quantized_dimension > dimension || group_size == 0 ||
      quantized_dimension % group_size != 0 || input.size() != elements ||
      output.size() != elements) {
    return Status::InvalidArgument("DeepSeek KV FP8 simulation is invalid");
  }
  std::vector<BFloat16> result(input.begin(), input.end());
  for (std::uint32_t token = 0; token < tokens; ++token)
    for (std::uint32_t group = 0; group < quantized_dimension / group_size;
         ++group) {
      const auto begin = static_cast<std::size_t>(token) * dimension +
                         static_cast<std::size_t>(group) * group_size;
      float amax = 0.0F;
      for (std::uint32_t offset = 0; offset < group_size; ++offset) {
        const auto value = input[begin + offset].to_float();
        if (!std::isfinite(value))
          return Status::InvalidArgument(
              "DeepSeek KV FP8 simulation input is non-finite");
        amax = std::max(amax, std::abs(value));
      }
      amax = std::max(amax, DeepSeekFp8ActivationCodec::kAmaxFloor);
      const auto exponent = static_cast<int>(std::ceil(std::log2(
          static_cast<double>(amax) / DeepSeekFp8ActivationCodec::kMaxFinite)));
      if (exponent < -127 || exponent > 127)
        return Status::InvalidArgument(
            "DeepSeek KV FP8 simulation scale is invalid");
      const auto scale = std::ldexp(1.0F, exponent);
      for (std::uint32_t offset = 0; offset < group_size; ++offset) {
        const auto normalized = std::clamp(
            input[begin + offset].to_float() / scale,
            -DeepSeekFp8ActivationCodec::kMaxFinite,
            DeepSeekFp8ActivationCodec::kMaxFinite);
        auto encoded = DeepSeekFp8ActivationCodec::EncodeE4m3Fn(normalized);
        if (!encoded.ok()) return encoded.status();
        auto decoded = DeepSeekFp8ActivationCodec::DecodeE4m3Fn(*encoded);
        if (!decoded.ok()) return decoded.status();
        result[begin + offset] = BFloat16::FromFloat(*decoded * scale);
      }
    }
  std::copy(result.begin(), result.end(), output.begin());
  return Status::Ok();
}

Status deepseek_grouped_bf16_gemm_oracle(
    std::span<const BFloat16> input, std::span<const BFloat16> weight,
    std::uint32_t tokens, std::uint32_t groups,
    std::uint32_t outputs, std::uint32_t inputs,
    std::span<BFloat16> output) {
  const auto input_elements = static_cast<std::size_t>(tokens) * groups * inputs;
  const auto weight_elements = static_cast<std::size_t>(groups) * outputs * inputs;
  const auto output_elements = static_cast<std::size_t>(tokens) * groups * outputs;
  if (tokens == 0 || groups == 0 || outputs == 0 || inputs == 0 ||
      input.size() != input_elements || weight.size() != weight_elements ||
      output.size() != output_elements) {
    return Status::InvalidArgument("DeepSeek grouped BF16 GEMM is invalid");
  }
  std::vector<BFloat16> result(output_elements);
  for (std::uint32_t token = 0; token < tokens; ++token)
    for (std::uint32_t group = 0; group < groups; ++group)
      for (std::uint32_t row = 0; row < outputs; ++row) {
        float sum = 0.0F;
        for (std::uint32_t column = 0; column < inputs; ++column) {
          const auto x = input[(static_cast<std::size_t>(token) * groups + group) *
                               inputs + column].to_float();
          const auto w = weight[(static_cast<std::size_t>(group) * outputs + row) *
                                inputs + column].to_float();
          if (!std::isfinite(x) || !std::isfinite(w))
            return Status::InvalidArgument(
                "DeepSeek grouped BF16 GEMM input is non-finite");
          sum += x * w;
        }
        result[(static_cast<std::size_t>(token) * groups + group) * outputs + row] =
            BFloat16::FromFloat(sum);
      }
  std::copy(result.begin(), result.end(), output.begin());
  return Status::Ok();
}

}  // namespace pih
