#include "pih/model/qwen3_gqa_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

constexpr std::uint32_t kQueryHeads = 16;
constexpr std::uint32_t kKvHeads = 8;
constexpr std::uint32_t kHeadDimension = 128;
constexpr std::uint32_t kMaximumQueryTokens = 4096;
constexpr std::uint32_t kMaximumHistoryTokens = 40960;

Result<std::size_t> tensor_extent(std::uint64_t tokens,
                                  std::uint64_t heads) {
  auto rows = checked_mul_u64(tokens, heads);
  if (!rows.ok()) return rows.status();
  auto elements = checked_mul_u64(rows.value(), kHeadDimension);
  if (!elements.ok()) return elements.status();
  if (elements.value() > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted("Qwen GQA extent exceeds address space");
  }
  return static_cast<std::size_t>(elements.value());
}

}  // namespace

Status qwen_gqa_attention_oracle(
    std::span<const BFloat16> query, std::span<const BFloat16> key_cache,
    std::span<const BFloat16> value_cache, std::uint32_t query_tokens,
    std::uint32_t history_tokens, std::uint32_t query_start_position,
    std::span<BFloat16> output) {
  if (query_tokens == 0 || query_tokens > kMaximumQueryTokens ||
      history_tokens == 0 || history_tokens > kMaximumHistoryTokens ||
      query_start_position >= history_tokens ||
      query_tokens > history_tokens - query_start_position) {
    return Status::InvalidArgument("Qwen GQA token range is invalid");
  }
  auto query_extent = tensor_extent(query_tokens, kQueryHeads);
  auto cache_extent = tensor_extent(history_tokens, kKvHeads);
  if (!query_extent.ok()) return query_extent.status();
  if (!cache_extent.ok()) return cache_extent.status();
  if (query.size() != query_extent.value() ||
      output.size() != query_extent.value() ||
      key_cache.size() != cache_extent.value() ||
      value_cache.size() != cache_extent.value()) {
    return Status::InvalidArgument("Qwen GQA tensor extent is invalid");
  }

  constexpr float kScale = 0.08838834764831845F;  // 1/sqrt(128)
  std::vector<float> result(output.size());
  for (std::uint32_t row = 0; row < query_tokens; ++row) {
    const std::uint32_t last_key = query_start_position + row;
    for (std::uint32_t query_head = 0; query_head < kQueryHeads;
         ++query_head) {
      const std::uint32_t kv_head = query_head / (kQueryHeads / kKvHeads);
      const std::size_t query_offset =
          (static_cast<std::size_t>(row) * kQueryHeads + query_head) *
          kHeadDimension;
      std::array<float, kHeadDimension> accumulator{};
      float maximum = -std::numeric_limits<float>::infinity();
      float denominator = 0.0F;
      for (std::uint32_t key_token = 0; key_token <= last_key; ++key_token) {
        const std::size_t cache_offset =
            (static_cast<std::size_t>(key_token) * kKvHeads + kv_head) *
            kHeadDimension;
        float score = 0.0F;
        for (std::size_t column = 0; column < kHeadDimension; ++column) {
          score += query[query_offset + column].to_float() *
                   key_cache[cache_offset + column].to_float();
        }
        score *= kScale;
        const float next_maximum = std::max(maximum, score);
        const float prior_scale = std::exp(maximum - next_maximum);
        const float value_scale = std::exp(score - next_maximum);
        denominator = denominator * prior_scale + value_scale;
        for (std::size_t column = 0; column < kHeadDimension; ++column) {
          accumulator[column] = accumulator[column] * prior_scale +
                                value_cache[cache_offset + column].to_float() *
                                    value_scale;
        }
        maximum = next_maximum;
      }
      if (!std::isfinite(denominator) || denominator <= 0.0F) {
        return Status::FailedPrecondition(
            "Qwen GQA online softmax produced an invalid denominator");
      }
      for (std::size_t column = 0; column < kHeadDimension; ++column) {
        const float value = accumulator[column] / denominator;
        if (!std::isfinite(value)) {
          return Status::FailedPrecondition(
              "Qwen GQA attention produced a nonfinite value");
        }
        result[query_offset + column] = value;
      }
    }
  }
  std::vector<BFloat16> converted;
  converted.reserve(result.size());
  for (const float value : result) converted.push_back(BFloat16::FromFloat(value));
  std::copy(converted.begin(), converted.end(), output.begin());
  return Status::Ok();
}

}  // namespace pih
