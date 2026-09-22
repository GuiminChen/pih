#include "pih/model/deepseek_mhc_oracle.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pih {
namespace {

constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kMixRows = 2 * kStreams + kStreams * kStreams;

bool finite(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

float sigmoid(float value) {
  if (value >= 0.0F) {
    const auto exponential = std::exp(-value);
    return 1.0F / (1.0F + exponential);
  }
  const auto exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

}  // namespace

Status deepseek_mhc_pre_oracle(
    std::span<const BFloat16> residual,
    const DeepSeekMhcParameters& parameters,
    std::uint32_t token_count, std::uint32_t hidden_size,
    std::span<float> post_mix, std::span<float> residual_mix,
    std::span<BFloat16> layer_input) {
  const auto residual_elements = static_cast<std::size_t>(token_count) *
                                 kStreams * hidden_size;
  const auto flat_width = static_cast<std::size_t>(kStreams) * hidden_size;
  if (token_count == 0 || token_count > 4096 || hidden_size == 0 ||
      hidden_size > 4096 || residual.size() != residual_elements ||
      parameters.fn.size() != kMixRows * flat_width ||
      parameters.scale.size() != 3 || parameters.base.size() != kMixRows ||
      post_mix.size() != static_cast<std::size_t>(token_count) * kStreams ||
      residual_mix.size() !=
          static_cast<std::size_t>(token_count) * kStreams * kStreams ||
      layer_input.size() !=
          static_cast<std::size_t>(token_count) * hidden_size ||
      !finite(parameters.fn) || !finite(parameters.scale) ||
      !finite(parameters.base) || !std::isfinite(parameters.rms_epsilon) ||
      !std::isfinite(parameters.pre_epsilon) ||
      !std::isfinite(parameters.sinkhorn_epsilon) ||
      !std::isfinite(parameters.post_multiplier) ||
      parameters.rms_epsilon <= 0.0F || parameters.pre_epsilon < 0.0F ||
      parameters.sinkhorn_epsilon < 0.0F ||
      parameters.post_multiplier <= 0.0F ||
      parameters.sinkhorn_iterations == 0 ||
      parameters.sinkhorn_iterations > 16) {
    return Status::InvalidArgument("DeepSeek mHC pre contract is invalid");
  }
  std::vector<float> next_post(post_mix.size());
  std::vector<float> next_mix(residual_mix.size());
  std::vector<BFloat16> next_input(layer_input.size());
  std::vector<float> logits(kMixRows);
  for (std::uint32_t token = 0; token < token_count; ++token) {
    const auto residual_base = static_cast<std::size_t>(token) * flat_width;
    float square_sum = 0.0F;
    for (std::size_t column = 0; column < flat_width; ++column) {
      const auto value = residual[residual_base + column].to_float();
      if (!std::isfinite(value)) {
        return Status::InvalidArgument("DeepSeek mHC residual is non-finite");
      }
      square_sum += value * value;
    }
    const auto inverse_rms = 1.0F / std::sqrt(
        square_sum / static_cast<float>(flat_width) +
        parameters.rms_epsilon);
    for (std::uint32_t row = 0; row < kMixRows; ++row) {
      float value = 0.0F;
      const auto fn_base = static_cast<std::size_t>(row) * flat_width;
      for (std::size_t column = 0; column < flat_width; ++column) {
        value += residual[residual_base + column].to_float() *
                 parameters.fn[fn_base + column];
      }
      logits[row] = value * inverse_rms;
    }
    const auto post_base = static_cast<std::size_t>(token) * kStreams;
    for (std::uint32_t stream = 0; stream < kStreams; ++stream) {
      logits[stream] = sigmoid(logits[stream] * parameters.scale[0] +
                               parameters.base[stream]) +
                       parameters.pre_epsilon;
      next_post[post_base + stream] =
          sigmoid(logits[kStreams + stream] * parameters.scale[1] +
                  parameters.base[kStreams + stream]) *
          parameters.post_multiplier;
    }
    const auto mix_base = static_cast<std::size_t>(token) *
                          kStreams * kStreams;
    for (std::uint32_t input = 0; input < kStreams; ++input) {
      float maximum = -INFINITY;
      for (std::uint32_t output = 0; output < kStreams; ++output) {
        const auto index = 2 * kStreams + input * kStreams + output;
        logits[index] = logits[index] * parameters.scale[2] +
                        parameters.base[index];
        maximum = std::max(maximum, logits[index]);
      }
      float sum = 0.0F;
      for (std::uint32_t output = 0; output < kStreams; ++output) {
        const auto index = 2 * kStreams + input * kStreams + output;
        const auto value = std::exp(logits[index] - maximum);
        next_mix[mix_base + input * kStreams + output] = value;
        sum += value;
      }
      for (std::uint32_t output = 0; output < kStreams; ++output) {
        next_mix[mix_base + input * kStreams + output] =
            next_mix[mix_base + input * kStreams + output] / sum +
            parameters.sinkhorn_epsilon;
      }
    }
    auto normalize_columns = [&] {
      for (std::uint32_t output = 0; output < kStreams; ++output) {
        float sum = 0.0F;
        for (std::uint32_t input = 0; input < kStreams; ++input) {
          sum += next_mix[mix_base + input * kStreams + output];
        }
        sum += parameters.sinkhorn_epsilon;
        for (std::uint32_t input = 0; input < kStreams; ++input) {
          next_mix[mix_base + input * kStreams + output] /= sum;
        }
      }
    };
    normalize_columns();
    for (std::uint32_t iteration = 1;
         iteration < parameters.sinkhorn_iterations; ++iteration) {
      for (std::uint32_t input = 0; input < kStreams; ++input) {
        float sum = parameters.sinkhorn_epsilon;
        for (std::uint32_t output = 0; output < kStreams; ++output) {
          sum += next_mix[mix_base + input * kStreams + output];
        }
        for (std::uint32_t output = 0; output < kStreams; ++output) {
          next_mix[mix_base + input * kStreams + output] /= sum;
        }
      }
      normalize_columns();
    }
    for (std::uint32_t column = 0; column < hidden_size; ++column) {
      float value = 0.0F;
      for (std::uint32_t stream = 0; stream < kStreams; ++stream) {
        value += logits[stream] *
                 residual[residual_base +
                          static_cast<std::size_t>(stream) * hidden_size +
                          column].to_float();
      }
      next_input[static_cast<std::size_t>(token) * hidden_size + column] =
          BFloat16::FromFloat(value);
    }
  }
  std::copy(next_post.begin(), next_post.end(), post_mix.begin());
  std::copy(next_mix.begin(), next_mix.end(), residual_mix.begin());
  std::copy(next_input.begin(), next_input.end(), layer_input.begin());
  return Status::Ok();
}

Status deepseek_mhc_post_oracle(
    std::span<const BFloat16> layer_output,
    std::span<const BFloat16> residual,
    std::span<const float> post_mix,
    std::span<const float> residual_mix,
    std::uint32_t token_count, std::uint32_t hidden_size,
    std::span<BFloat16> output) {
  const auto hidden_elements = static_cast<std::size_t>(token_count) * hidden_size;
  const auto residual_elements = hidden_elements * kStreams;
  if (token_count == 0 || token_count > 4096 || hidden_size == 0 ||
      hidden_size > 4096 || layer_output.size() != hidden_elements ||
      residual.size() != residual_elements || output.size() != residual_elements ||
      post_mix.size() != static_cast<std::size_t>(token_count) * kStreams ||
      residual_mix.size() !=
          static_cast<std::size_t>(token_count) * kStreams * kStreams ||
      !finite(post_mix) || !finite(residual_mix)) {
    return Status::InvalidArgument("DeepSeek mHC post contract is invalid");
  }
  std::vector<BFloat16> next(output.size());
  for (std::uint32_t token = 0; token < token_count; ++token) {
    const auto hidden_base = static_cast<std::size_t>(token) * hidden_size;
    const auto residual_base = hidden_base * kStreams;
    const auto mix_base = static_cast<std::size_t>(token) * kStreams * kStreams;
    for (std::uint32_t target = 0; target < kStreams; ++target) {
      for (std::uint32_t column = 0; column < hidden_size; ++column) {
        const auto layer = layer_output[hidden_base + column].to_float();
        if (!std::isfinite(layer)) {
          return Status::InvalidArgument("DeepSeek mHC output is non-finite");
        }
        float value = post_mix[token * kStreams + target] * layer;
        for (std::uint32_t source = 0; source < kStreams; ++source) {
          const auto residual_value =
              residual[residual_base +
                       static_cast<std::size_t>(source) * hidden_size +
                       column].to_float();
          if (!std::isfinite(residual_value)) {
            return Status::InvalidArgument(
                "DeepSeek mHC residual is non-finite");
          }
          value += residual_mix[mix_base + source * kStreams + target] *
                   residual_value;
        }
        next[residual_base + static_cast<std::size_t>(target) * hidden_size +
             column] = BFloat16::FromFloat(value);
      }
    }
  }
  std::copy(next.begin(), next.end(), output.begin());
  return Status::Ok();
}

}  // namespace pih
