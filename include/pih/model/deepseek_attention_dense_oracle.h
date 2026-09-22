#pragma once

#include <span>

#include "pih/core/bfloat16.h"
#include "pih/core/status.h"

namespace pih {
Status deepseek_head_rms_oracle(
    std::span<const BFloat16> input, std::uint32_t token_count,
    std::uint32_t head_count, std::uint32_t head_dimension, float epsilon,
    std::span<BFloat16> output);
Status deepseek_rotary_oracle(
    std::span<const BFloat16> input, std::span<const float> frequencies,
    std::uint32_t token_count, std::uint32_t head_count,
    std::uint32_t head_dimension, std::uint32_t rope_dimension, bool inverse,
    std::span<BFloat16> output);
Status deepseek_kv_fp8_simulate_oracle(
    std::span<const BFloat16> input, std::uint32_t token_count,
    std::uint32_t vector_dimension, std::uint32_t quantized_dimension,
    std::uint32_t group_size, std::span<BFloat16> output);
Status deepseek_grouped_bf16_gemm_oracle(
    std::span<const BFloat16> input, std::span<const BFloat16> weight,
    std::uint32_t token_count, std::uint32_t group_count,
    std::uint32_t output_per_group, std::uint32_t input_per_group,
    std::span<BFloat16> output);
}  // namespace pih
