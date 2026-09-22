#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_execution_arena.h"
#include "pih/model/qwen3_bf16_packed_kv_metadata.h"
#include "pih/model/qwen3_sampler.h"

namespace pih {

struct Qwen3PackedSamplingWire final {
  std::uint32_t mode = 0;
  float temperature = 0.0F;
  float top_p = 1.0F;
  std::uint32_t top_k = 0;
  std::uint64_t seed = 0;
  std::uint64_t sample_ordinal = 0;
  std::uint32_t top_logprobs_count = 0;
  std::array<std::uint32_t,
             Qwen3SamplingDescriptor::kMaximumSuppressedTokenIds>
      suppressed_token_ids{};
  std::uint32_t suppressed_token_count = 0;
};
static_assert(sizeof(Qwen3PackedSamplingWire) == 112);

class QwenBf16PackedStepStagingLayout final {
 public:
  static constexpr std::uint64_t kAlignment = 256;
  static constexpr std::size_t kCopySpanCount = 14;

  static Result<QwenBf16PackedStepStagingLayout> Create(
      PackedTokenMetadataView token_metadata,
      QwenBf16PackedKvMetadataView kv_metadata);
  static Result<QwenBf16PackedStepStagingLayout> CreateBounded(
      std::uint32_t execution_bucket_tokens, std::uint32_t sequence_count,
      std::uint32_t visible_handle_count);

  Status materialize(PackedTokenMetadataView token_metadata,
                     QwenBf16PackedKvMetadataView kv_metadata,
                     std::span<std::byte> backing) const;
  Status materialize(
      PackedTokenMetadataView token_metadata,
      QwenBf16PackedKvMetadataView kv_metadata,
      std::span<const Qwen3SamplingDescriptor> sampling,
      std::span<std::byte> backing) const;

  [[nodiscard]] QwenBf16ArenaSpan token_ids() const noexcept { return spans_[0]; }
  [[nodiscard]] QwenBf16ArenaSpan positions() const noexcept { return spans_[1]; }
  [[nodiscard]] QwenBf16ArenaSpan request_index() const noexcept { return spans_[2]; }
  [[nodiscard]] QwenBf16ArenaSpan query_start_offsets() const noexcept { return spans_[3]; }
  [[nodiscard]] QwenBf16ArenaSpan sample_row_index() const noexcept { return spans_[4]; }
  [[nodiscard]] QwenBf16ArenaSpan sample_count() const noexcept { return spans_[5]; }
  [[nodiscard]] QwenBf16ArenaSpan append_handles() const noexcept { return spans_[6]; }
  [[nodiscard]] QwenBf16ArenaSpan token_offsets() const noexcept { return spans_[7]; }
  [[nodiscard]] QwenBf16ArenaSpan visible_handle_offsets() const noexcept { return spans_[8]; }
  [[nodiscard]] QwenBf16ArenaSpan visible_handles() const noexcept { return spans_[9]; }
  [[nodiscard]] QwenBf16ArenaSpan key_token_counts() const noexcept { return spans_[10]; }
  [[nodiscard]] QwenBf16ArenaSpan owner_sequence_indices() const noexcept { return spans_[11]; }
  [[nodiscard]] QwenBf16ArenaSpan sampling_descriptors() const noexcept {
    return spans_[12];
  }
  [[nodiscard]] QwenBf16ArenaSpan sample_sequence_indices() const noexcept {
    return spans_[13];
  }
  [[nodiscard]] const std::array<QwenBf16ArenaSpan, kCopySpanCount>&
  copy_spans() const noexcept { return spans_; }
  [[nodiscard]] std::uint64_t total_bytes() const noexcept { return total_bytes_; }
  [[nodiscard]] std::uint32_t execution_bucket_tokens() const noexcept {
    return execution_bucket_tokens_;
  }
  [[nodiscard]] std::uint32_t sequence_count() const noexcept {
    return sequence_count_;
  }
  [[nodiscard]] std::uint32_t visible_handle_count() const noexcept {
    return visible_handle_count_;
  }

 private:
  std::array<QwenBf16ArenaSpan, kCopySpanCount> spans_{};
  std::uint64_t total_bytes_ = 0;
  std::uint32_t execution_bucket_tokens_ = 0;
  std::uint32_t sequence_count_ = 0;
  std::uint32_t visible_handle_count_ = 0;
};

}  // namespace pih
