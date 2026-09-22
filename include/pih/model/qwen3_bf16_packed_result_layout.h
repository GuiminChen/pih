#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_bf16_execution_arena.h"

namespace pih {

struct QwenBf16PackedSampleReceipt final {
  std::uint32_t token_id = 0;
  float selected_logprob = 0.0F;
  std::uint32_t rng_word = 0;
  std::uint32_t top_logprob_count = 0;
  std::array<std::uint32_t, 20> top_token_ids{};
  std::array<float, 20> top_logprobs{};
};

class QwenBf16PackedResultLayout final {
 public:
  static constexpr std::uint64_t kAlignment = 256;
  static constexpr std::uint32_t kMaximumSamples = 4096;
  static constexpr std::uint32_t kVocabularySize = 151936;

  static Result<QwenBf16PackedResultLayout> Create(
      std::uint32_t sample_capacity);

  Status initialize(std::span<std::byte> backing) const;
  Result<std::vector<std::uint32_t>> parse(
      std::span<const std::byte> backing, std::uint32_t sample_count,
      bool publication_authorized) const;
  Result<std::vector<QwenBf16PackedSampleReceipt>> parse_sampling(
      std::span<const std::byte> backing, std::uint32_t sample_count,
      bool publication_authorized) const;

  [[nodiscard]] std::uint32_t sample_capacity() const noexcept {
    return sample_capacity_;
  }
  [[nodiscard]] QwenBf16ArenaSpan sampled_token_ids() const noexcept {
    return sampled_token_ids_;
  }
  [[nodiscard]] QwenBf16ArenaSpan selected_logprobs() const noexcept {
    return selected_logprobs_;
  }
  [[nodiscard]] QwenBf16ArenaSpan rng_words() const noexcept {
    return rng_words_;
  }
  [[nodiscard]] QwenBf16ArenaSpan top_logprob_token_ids() const noexcept {
    return top_logprob_token_ids_;
  }
  [[nodiscard]] QwenBf16ArenaSpan top_logprobs() const noexcept {
    return top_logprobs_;
  }
  [[nodiscard]] QwenBf16ArenaSpan top_logprob_counts() const noexcept {
    return top_logprob_counts_;
  }
  [[nodiscard]] QwenBf16ArenaSpan device_error() const noexcept {
    return device_error_;
  }
  [[nodiscard]] std::uint64_t total_bytes() const noexcept {
    return total_bytes_;
  }

 private:
  QwenBf16PackedResultLayout(std::uint32_t sample_capacity,
                             QwenBf16ArenaSpan sampled_token_ids,
                             QwenBf16ArenaSpan selected_logprobs,
                             QwenBf16ArenaSpan rng_words,
                             QwenBf16ArenaSpan top_logprob_token_ids,
                             QwenBf16ArenaSpan top_logprobs,
                             QwenBf16ArenaSpan top_logprob_counts,
                             QwenBf16ArenaSpan device_error,
                             std::uint64_t total_bytes)
      : sample_capacity_(sample_capacity),
        sampled_token_ids_(sampled_token_ids),
        selected_logprobs_(selected_logprobs), rng_words_(rng_words),
        top_logprob_token_ids_(top_logprob_token_ids),
        top_logprobs_(top_logprobs), top_logprob_counts_(top_logprob_counts),
        device_error_(device_error),
        total_bytes_(total_bytes) {}

  std::uint32_t sample_capacity_ = 0;
  QwenBf16ArenaSpan sampled_token_ids_{};
  QwenBf16ArenaSpan selected_logprobs_{};
  QwenBf16ArenaSpan rng_words_{};
  QwenBf16ArenaSpan top_logprob_token_ids_{};
  QwenBf16ArenaSpan top_logprobs_{};
  QwenBf16ArenaSpan top_logprob_counts_{};
  QwenBf16ArenaSpan device_error_{};
  std::uint64_t total_bytes_ = 0;
};

}  // namespace pih
