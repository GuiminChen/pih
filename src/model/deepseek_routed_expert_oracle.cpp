#include "pih/model/deepseek_routed_expert_oracle.h"

#include <cmath>

#include "pih/core/checked_math.h"
#include "pih/model/deepseek_expert_swiglu_oracle.h"
#include "pih/model/deepseek_fp4_gemm_oracle.h"
#include "pih/model/deepseek_fp8_activation_codec.h"

namespace pih {
namespace {

std::vector<float> to_float(std::span<const BFloat16> values) {
  std::vector<float> result;
  result.reserve(values.size());
  for (const auto value : values) result.push_back(value.to_float());
  return result;
}

Result<std::vector<float>> rounded_bf16(std::span<const float> values) {
  std::vector<float> result;
  result.reserve(values.size());
  for (const auto value : values) {
    const auto rounded = BFloat16::FromFloat(value).to_float();
    if (!std::isfinite(rounded)) {
      return Status::InvalidArgument(
          "DeepSeek expert BF16 boundary is nonfinite");
    }
    result.push_back(rounded);
  }
  return result;
}

Result<std::vector<float>> project(std::span<const float> input,
                                   std::uint32_t tokens,
                                   std::uint32_t input_size,
                                   std::uint32_t output_size,
                                   DeepSeekFp4MatrixView matrix) {
  auto quantized =
      DeepSeekFp8ActivationCodec::Quantize(input, tokens, input_size);
  if (!quantized.ok()) return quantized.status();
  auto projected = DeepSeekFp4GemmOracle::Multiply(
      *quantized, matrix.packed, matrix.scale_bits, output_size);
  if (!projected.ok()) return projected.status();
  return rounded_bf16(*projected);
}

}  // namespace

Result<std::vector<BFloat16>> DeepSeekRoutedExpertOracle::Forward(
    std::span<const BFloat16> input, std::span<const float> route_weights,
    std::uint32_t token_count, std::uint32_t hidden_size,
    std::uint32_t intermediate_size, DeepSeekFp4MatrixView w1,
    DeepSeekFp4MatrixView w2, DeepSeekFp4MatrixView w3) {
  auto input_elements = checked_mul_u64(token_count, hidden_size);
  if (token_count == 0 || hidden_size == 0 || intermediate_size == 0 ||
      hidden_size % DeepSeekFp8ActivationCodec::kValuesPerScale != 0 ||
      intermediate_size % DeepSeekFp8ActivationCodec::kValuesPerScale != 0 ||
      !input_elements.ok() || input.size() != *input_elements ||
      route_weights.size() != token_count) {
    return Status::InvalidArgument("DeepSeek routed expert shape is invalid");
  }
  for (const auto weight : route_weights) {
    if (!std::isfinite(weight) || weight < 0.0F) {
      return Status::InvalidArgument(
          "DeepSeek routed expert weight is invalid");
    }
  }

  const auto input_float = to_float(input);
  auto gate = project(input_float, token_count, hidden_size, intermediate_size,
                      w1);
  if (!gate.ok()) return gate.status();
  auto up = project(input_float, token_count, hidden_size, intermediate_size,
                    w3);
  if (!up.ok()) return up.status();
  auto middle = DeepSeekExpertSwiGluOracle::Apply(
      *gate, *up, token_count, intermediate_size);
  if (!middle.ok()) return middle.status();
  for (std::uint32_t token = 0; token < token_count; ++token) {
    for (std::uint32_t column = 0; column < intermediate_size; ++column) {
      (*middle)[static_cast<std::size_t>(token) * intermediate_size + column] *=
          route_weights[token];
    }
  }
  auto w2_input = rounded_bf16(*middle);
  if (!w2_input.ok()) return w2_input.status();
  auto output = project(*w2_input, token_count, intermediate_size, hidden_size,
                        w2);
  if (!output.ok()) return output.status();
  std::vector<BFloat16> result;
  result.reserve(output->size());
  for (const auto value : *output) result.push_back(BFloat16::FromFloat(value));
  return result;
}

}  // namespace pih
