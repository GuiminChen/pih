#pragma once

#include <cstdint>
#include <span>

#include "pih/core/bfloat16.h"
#include "pih/core/result.h"
#include "pih/core/status.h"

namespace pih {

Status deepseek_embedding_expand_oracle(
    std::span<const std::uint32_t> token_ids,
    std::span<const BFloat16> weight, std::uint32_t vocab_size,
    std::uint32_t hidden_size, std::span<BFloat16> output_hc);
Status deepseek_hc_head_oracle(
    std::span<const BFloat16> input_hc, std::span<const float> fn,
    std::span<const float> scale, std::span<const float> base,
    std::uint32_t token_count, std::uint32_t hidden_size,
    float rms_epsilon, float hc_epsilon, std::span<BFloat16> output);
Status deepseek_lm_head_oracle(
    std::span<const BFloat16> input, std::span<const BFloat16> weight,
    std::uint32_t output_rows, std::uint32_t vocab_size,
    std::uint32_t hidden_size, std::span<float> logits);
Status deepseek_dspark_markov_oracle(
    std::span<const std::uint32_t> token_ids,
    std::span<const BFloat16> embedding_weight,
    std::span<const BFloat16> head_weight,
    std::span<const float> raw_logits, std::uint32_t vocab_size,
    std::uint32_t markov_rank, std::span<BFloat16> markov_embeddings,
    std::span<float> biased_logits);
Status deepseek_dspark_confidence_oracle(
    std::span<const BFloat16> hidden,
    std::span<const BFloat16> markov_embeddings,
    std::span<const float> projection_weight, std::uint32_t row_count,
    std::uint32_t hidden_size, std::uint32_t markov_rank,
    std::span<float> confidence);
Result<std::uint32_t> deepseek_argmax_oracle(
    std::span<const float> logits);
Status deepseek_dspark_draft_init_oracle(
    std::span<const std::uint32_t> input_token_ids,
    std::span<const BFloat16> embedding_weight,
    std::uint32_t noise_token_id, std::uint32_t block_size,
    std::uint32_t vocab_size, std::uint32_t hidden_size,
    std::span<std::uint32_t> draft_token_ids,
    std::span<BFloat16> output_hc);

}  // namespace pih
