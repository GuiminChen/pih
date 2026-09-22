#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/sha256.h"

namespace pih {

struct DeepSeekChunkedPrefillSequence final {
  std::uint64_t request_id = 0;
  std::uint64_t request_generation = 0;
  std::uint32_t prompt_token_count = 0;
  std::uint32_t consumed_token_count = 0;
};

struct DeepSeekChunkedPrefillSlice final {
  std::uint32_t sequence_index = 0;
  std::uint64_t request_id = 0;
  std::uint64_t request_generation = 0;
  std::uint32_t start_position = 0;
  std::uint32_t token_count = 0;
};

class DeepSeekChunkedPrefillPlan final {
 public:
  static Result<DeepSeekChunkedPrefillPlan> Compile(
      std::uint64_t plan_sequence, std::uint32_t maximum_chunk_tokens,
      std::span<const DeepSeekChunkedPrefillSequence> sequences);

  [[nodiscard]] std::uint64_t plan_sequence() const noexcept {
    return plan_sequence_;
  }
  [[nodiscard]] std::span<const DeepSeekChunkedPrefillSlice> slices()
      const noexcept { return slices_; }
  [[nodiscard]] std::uint32_t token_count() const noexcept {
    return token_count_;
  }
  [[nodiscard]] bool publishes_sampled_token() const noexcept {
    return publishes_sampled_token_;
  }
  [[nodiscard]] const Sha256Digest& identity() const noexcept {
    return identity_;
  }

 private:
  std::uint64_t plan_sequence_ = 0;
  std::vector<DeepSeekChunkedPrefillSlice> slices_;
  std::uint32_t token_count_ = 0;
  bool publishes_sampled_token_ = false;
  Sha256Digest identity_{};
};

}  // namespace pih
