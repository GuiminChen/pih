#include "pih/model/deepseek_rms_norm_oracle.h"

#include <cmath>
#include <vector>

namespace pih {

Status deepseek_rms_norm_oracle(
    std::span<const BFloat16> input, std::span<const BFloat16> weight,
    std::uint32_t rows, std::uint32_t hidden_size,
    std::span<BFloat16> output) {
  if (rows == 0 || (hidden_size != 512 && hidden_size != 1024 &&
                    hidden_size != 4096)) {
    return Status::InvalidArgument("DeepSeek RMSNorm shape is invalid");
  }
  const auto elements = static_cast<std::size_t>(rows) * hidden_size;
  if (input.size() != elements || output.size() != elements ||
      weight.size() != hidden_size) {
    return Status::InvalidArgument("DeepSeek RMSNorm extent is invalid");
  }
  std::vector<float> result(elements);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const auto offset = static_cast<std::size_t>(row) * hidden_size;
    float sum = 0.0F;
    for (std::uint32_t column = 0; column < hidden_size; ++column) {
      const auto value = input[offset + column].to_float();
      const auto scale = weight[column].to_float();
      if (!std::isfinite(value) || !std::isfinite(scale)) {
        return Status::InvalidArgument("DeepSeek RMSNorm input is non-finite");
      }
      sum += value * value;
    }
    const auto inverse_rms = 1.0F / std::sqrt(
        sum / static_cast<float>(hidden_size) + 0.000001F);
    for (std::uint32_t column = 0; column < hidden_size; ++column) {
      result[offset + column] = input[offset + column].to_float() *
                                inverse_rms * weight[column].to_float();
    }
  }
  for (std::size_t index = 0; index < elements; ++index) {
    output[index] = BFloat16::FromFloat(result[index]);
  }
  return Status::Ok();
}

}  // namespace pih
