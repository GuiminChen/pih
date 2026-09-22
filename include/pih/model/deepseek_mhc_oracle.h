#pragma once

#include <cstdint>
#include <span>

#include "pih/core/bfloat16.h"
#include "pih/core/status.h"

namespace pih {

struct DeepSeekMhcParameters final {
  std::span<const float> fn;
  std::span<const float> scale;
  std::span<const float> base;
  float rms_epsilon = 0.0F;
  float pre_epsilon = 0.0F;
  float sinkhorn_epsilon = 0.0F;
  float post_multiplier = 0.0F;
  std::uint32_t sinkhorn_iterations = 0;
};

Status deepseek_mhc_pre_oracle(
    std::span<const BFloat16> residual,
    const DeepSeekMhcParameters& parameters,
    std::uint32_t token_count, std::uint32_t hidden_size,
    std::span<float> post_mix, std::span<float> residual_mix,
    std::span<BFloat16> layer_input);

Status deepseek_mhc_post_oracle(
    std::span<const BFloat16> layer_output,
    std::span<const BFloat16> residual,
    std::span<const float> post_mix,
    std::span<const float> residual_mix,
    std::uint32_t token_count, std::uint32_t hidden_size,
    std::span<BFloat16> output);

}  // namespace pih
