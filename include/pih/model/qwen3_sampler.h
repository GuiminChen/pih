#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

enum class Qwen3SamplingMode : std::uint8_t {
  kGreedy,
  kStochastic,
};

struct Qwen3SamplingDescriptor final {
  static constexpr std::uint32_t kMaximumSuppressedTokenIds = 17;
  Qwen3SamplingMode mode = Qwen3SamplingMode::kGreedy;
  float temperature = 0.0F;
  float top_p = 1.0F;
  std::optional<std::uint32_t> top_k;
  std::uint64_t seed = 0;
  std::uint64_t sample_ordinal = 0;
  std::uint32_t top_logprobs_count = 0;
  std::array<std::uint32_t, kMaximumSuppressedTokenIds>
      suppressed_token_ids{};
  std::uint32_t suppressed_token_count = 0;
};

struct Qwen3SamplingResult final {
  std::uint32_t token_id = 0;
  float selected_logprob = 0.0F;
  std::uint32_t rng_word = 0;
  std::vector<std::uint32_t> top_token_ids;
  std::vector<float> top_logprobs;
  bool operator==(const Qwen3SamplingResult&) const = default;
};

Result<std::uint32_t> qwen3_philox_word(std::uint64_t seed,
                                       std::uint64_t sample_ordinal);
Result<Qwen3SamplingResult> qwen3_sample_cpu(
    std::span<const float> logits,
    const Qwen3SamplingDescriptor& descriptor);

}  // namespace pih
