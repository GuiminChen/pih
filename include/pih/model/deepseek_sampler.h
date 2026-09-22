#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <optional>
#include <vector>

#include "pih/core/result.h"
#include "pih/model/deepseek_token_suppression.h"

namespace pih {

enum class DeepSeekSamplingMode : std::uint8_t {
  kGreedy,
  kStochastic,
};

struct DeepSeekRequestSamplingConfig final {
  static constexpr std::uint32_t kMaximumStopTokenIds = 16;
  std::uint64_t config_id = 0;
  DeepSeekSamplingMode mode = DeepSeekSamplingMode::kGreedy;
  std::uint64_t effective_seed = 0;
  float temperature = 0.0F;
  float top_p = 1.0F;
  std::optional<std::uint32_t> top_k;
  bool logprobs_enabled = false;
  std::uint32_t top_logprobs_count = 0;
  std::array<std::uint32_t, kMaximumStopTokenIds> stop_token_ids{};
  std::uint32_t stop_token_count = 0;
};

// Canonical request-local pih-sampler-v1 RNG word. The caller advances
// ordinal only when the corresponding accepted-token record is committed.
Result<std::uint32_t> deepseek_philox_word(std::uint64_t seed,
                                           std::uint64_t sample_ordinal);
Result<std::uint32_t> deepseek_greedy_sample(
    std::span<const float> logits,
    const DeepSeekSuppressedTokenSet& suppressed_tokens = {});

struct DeepSeekSamplingDescriptor final {
  float temperature = 1.0F;
  float top_p = 1.0F;
  std::optional<std::uint32_t> top_k;
  std::uint64_t seed = 0;
  std::uint64_t sample_ordinal = 0;
  bool logprobs_enabled = false;
  std::uint32_t top_logprobs_count = 0;
  DeepSeekSuppressedTokenSet suppressed_tokens;
};

struct DeepSeekPreparedSamplingInput final {
  std::uint64_t config_id = 0;
  DeepSeekSamplingMode mode = DeepSeekSamplingMode::kGreedy;
  DeepSeekSamplingDescriptor descriptor;
};

struct DeepSeekSamplingResult final {
  std::uint32_t token_id = 0;
  float selected_logprob = 0.0F;
  std::uint32_t rng_word = 0;
  std::vector<std::uint32_t> top_token_ids;
  std::vector<float> top_logprobs;
};

Result<DeepSeekSamplingResult> deepseek_sample_cpu(
    std::span<const float> logits,
    const DeepSeekSamplingDescriptor& descriptor);

}  // namespace pih
