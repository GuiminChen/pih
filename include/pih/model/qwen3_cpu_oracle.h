#pragma once

#include <cstdint>
#include <span>

#include "pih/core/bfloat16.h"
#include "pih/core/status.h"
#include "pih/core/result.h"

namespace pih {

Status qwen_embedding_oracle(std::span<const BFloat16> table,
                             std::uint64_t vocabulary_size,
                             std::uint64_t hidden_size,
                             std::span<const std::int64_t> token_ids,
                             std::span<BFloat16> output);

Status qwen_rms_norm_oracle(std::span<const BFloat16> input,
                            std::span<const BFloat16> weight,
                            std::uint64_t rows, std::uint64_t hidden_size,
                            float epsilon, std::span<BFloat16> output);

Status qwen_residual_add_oracle(std::span<const BFloat16> lhs,
                                std::span<const BFloat16> rhs,
                                std::span<BFloat16> output);

Status qwen_silu_mul_oracle(std::span<const BFloat16> gate,
                            std::span<const BFloat16> up,
                            std::span<BFloat16> output);

// Produces one shared FP32 angle pair per token and rotary dimension. Heads
// broadcast these rows; angle storage is never duplicated per attention head.
Status qwen_rope_angles_oracle(std::span<const std::int64_t> positions,
                               std::uint64_t head_dim, double theta,
                               std::span<float> cosine,
                               std::span<float> sine);

// Qwen rotate-half convention: [x0..xH, y0..yH] becomes
// [-y0..-yH, x0..xH]. cos/sin contain one value per pair and vector.
Status qwen_rope_oracle(std::span<const BFloat16> input,
                        std::span<const float> cosine,
                        std::span<const float> sine,
                        std::uint64_t tokens, std::uint64_t heads,
                        std::uint64_t head_dim,
                        std::span<BFloat16> output);

Result<std::int64_t> qwen_greedy_argmax_oracle(
    std::span<const float> logits);

}  // namespace pih
