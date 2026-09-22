#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "pih/model/qwen3_bf16_packed_batch_transaction.h"

namespace pih {

inline constexpr QwenKvBlockHandle kInvalidPackedKvBlockHandle{
    std::numeric_limits<std::uint32_t>::max(), 0};

struct QwenBf16PackedKvMetadataLimits final {
  std::uint32_t maximum_sequences;
  std::uint32_t maximum_execution_bucket_tokens;
  std::uint32_t maximum_visible_handles;
};

struct QwenBf16PackedKvMetadataView final {
  std::uint64_t generation;
  std::span<const QwenKvBlockHandle> append_handles;
  std::span<const std::uint16_t> token_offsets;
  std::span<const std::uint32_t> visible_handle_offsets;
  std::span<const QwenKvBlockHandle> visible_handles;
  std::span<const std::uint32_t> key_token_counts;
  std::span<const std::uint32_t> owner_sequence_indices;
};

class QwenBf16PackedKvMetadataArena final {
 public:
  static Result<QwenBf16PackedKvMetadataArena> Create(
      QwenBf16PackedKvMetadataLimits limits);

  Result<QwenBf16PackedKvMetadataView> materialize(
      const PackedTokenPlan& plan, PackedTokenMetadataView token_metadata,
      std::span<const QwenBf16PackedKvBinding> bindings,
      std::span<const QwenKvAppendPlan> append_plans);

 private:
  explicit QwenBf16PackedKvMetadataArena(
      QwenBf16PackedKvMetadataLimits limits)
      : limits_(limits),
        append_handles_(limits.maximum_execution_bucket_tokens,
                        kInvalidPackedKvBlockHandle),
        token_offsets_(limits.maximum_execution_bucket_tokens),
        visible_handle_offsets_(
            static_cast<std::size_t>(limits.maximum_sequences) + 1),
        visible_handles_(limits.maximum_visible_handles),
        key_token_counts_(limits.maximum_sequences),
        owner_sequence_indices_(limits.maximum_sequences) {}

  QwenBf16PackedKvMetadataLimits limits_{};
  std::vector<QwenKvBlockHandle> append_handles_;
  std::vector<std::uint16_t> token_offsets_;
  std::vector<std::uint32_t> visible_handle_offsets_;
  std::vector<QwenKvBlockHandle> visible_handles_;
  std::vector<std::uint32_t> key_token_counts_;
  std::vector<std::uint32_t> owner_sequence_indices_;
  std::uint64_t generation_ = 0;
};

}  // namespace pih
