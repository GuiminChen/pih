#pragma once

#include <cstdint>
#include <span>

#include "pih/core/bfloat16.h"
#include "pih/core/status.h"

namespace pih {

// Q: [query_tokens, 16, 128], K/V: [history_tokens, 8, 128].
// Each pair of query heads shares one KV head. The implementation maintains
// only one 128-float accumulator per row/head and never materializes scores.
Status qwen_gqa_attention_oracle(
    std::span<const BFloat16> query, std::span<const BFloat16> key_cache,
    std::span<const BFloat16> value_cache, std::uint32_t query_tokens,
    std::uint32_t history_tokens, std::uint32_t query_start_position,
    std::span<BFloat16> output);

}  // namespace pih
