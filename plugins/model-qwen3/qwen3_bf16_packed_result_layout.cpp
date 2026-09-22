#include "pih/model/qwen3_bf16_packed_result_layout.h"

#include <cstring>
#include <cmath>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_cuda_invariant.h"

namespace pih {

Result<QwenBf16PackedResultLayout> QwenBf16PackedResultLayout::Create(
    std::uint32_t sample_capacity) {
  if (sample_capacity == 0 || sample_capacity > kMaximumSamples) {
    return Status::InvalidArgument("packed result sample capacity is invalid");
  }
  auto token_bytes = checked_mul_u64(sample_capacity, sizeof(std::uint32_t));
  if (!token_bytes.ok()) return token_bytes.status();
  std::uint64_t cursor = *token_bytes;
  const auto append = [&cursor](std::uint64_t bytes)
      -> Result<QwenBf16ArenaSpan> {
    auto offset = checked_align_up_u64(cursor, kAlignment);
    if (!offset.ok()) return offset.status();
    auto end = checked_add_u64(*offset, bytes);
    if (!end.ok()) return end.status();
    cursor = *end;
    return QwenBf16ArenaSpan{*offset, bytes};
  };
  auto selected = append(*token_bytes);
  if (!selected.ok()) return selected.status();
  auto rng = append(*token_bytes);
  if (!rng.ok()) return rng.status();
  auto top_bytes = checked_mul_u64(*token_bytes, 20U);
  if (!top_bytes.ok()) return top_bytes.status();
  auto top_ids = append(*top_bytes);
  if (!top_ids.ok()) return top_ids.status();
  auto top_values = append(*top_bytes);
  if (!top_values.ok()) return top_values.status();
  auto top_counts = append(*token_bytes);
  if (!top_counts.ok()) return top_counts.status();
  auto error_offset = checked_align_up_u64(cursor, kAlignment);
  if (!error_offset.ok()) return error_offset.status();
  auto error_end = checked_add_u64(*error_offset, sizeof(std::uint32_t));
  if (!error_end.ok()) return error_end.status();
  auto total = checked_align_up_u64(*error_end, kAlignment);
  if (!total.ok()) return total.status();
  return QwenBf16PackedResultLayout(
      sample_capacity, {0, *token_bytes},
      *selected, *rng, *top_ids, *top_values, *top_counts,
      {*error_offset, sizeof(std::uint32_t)}, *total);
}

Result<std::vector<QwenBf16PackedSampleReceipt>>
QwenBf16PackedResultLayout::parse_sampling(
    std::span<const std::byte> backing, std::uint32_t sample_count,
    bool publication_authorized) const {
  auto tokens = parse(backing, sample_count, publication_authorized);
  if (!tokens.ok()) return tokens.status();
  std::vector<QwenBf16PackedSampleReceipt> receipts(sample_count);
  for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
    auto& receipt = receipts[sample];
    receipt.token_id = (*tokens)[sample];
    std::memcpy(&receipt.selected_logprob,
                backing.data() + selected_logprobs_.offset_bytes +
                    sample * sizeof(float),
                sizeof(float));
    std::memcpy(&receipt.rng_word,
                backing.data() + rng_words_.offset_bytes +
                    sample * sizeof(std::uint32_t),
                sizeof(std::uint32_t));
    std::memcpy(&receipt.top_logprob_count,
                backing.data() + top_logprob_counts_.offset_bytes +
                    sample * sizeof(std::uint32_t),
                sizeof(std::uint32_t));
    if (!std::isfinite(receipt.selected_logprob) ||
        receipt.top_logprob_count > 20U) {
      return Status::Internal("packed sampling receipt header is invalid");
    }
    for (std::uint32_t rank = 0; rank < receipt.top_logprob_count; ++rank) {
      const auto index = static_cast<std::uint64_t>(sample) * 20U + rank;
      std::memcpy(&receipt.top_token_ids[rank],
                  backing.data() + top_logprob_token_ids_.offset_bytes +
                      index * sizeof(std::uint32_t),
                  sizeof(std::uint32_t));
      std::memcpy(&receipt.top_logprobs[rank],
                  backing.data() + top_logprobs_.offset_bytes +
                      index * sizeof(float),
                  sizeof(float));
      if (receipt.top_token_ids[rank] >= kVocabularySize ||
          !std::isfinite(receipt.top_logprobs[rank])) {
        return Status::Internal("packed top-logprob receipt is invalid");
      }
    }
  }
  return receipts;
}

Status QwenBf16PackedResultLayout::initialize(
    std::span<std::byte> backing) const {
  if (backing.size() != total_bytes_) {
    return Status::InvalidArgument("packed result backing is invalid");
  }
  std::memset(backing.data(), 0xff, backing.size());
  return Status::Ok();
}

Result<std::vector<std::uint32_t>> QwenBf16PackedResultLayout::parse(
    std::span<const std::byte> backing, std::uint32_t sample_count,
    bool publication_authorized) const {
  if (!publication_authorized || backing.size() != total_bytes_ ||
      sample_count > sample_capacity_) {
    return Status::FailedPrecondition(
        "packed result is not authorized for publication");
  }
  std::uint32_t error = UINT32_MAX;
  std::memcpy(&error, backing.data() + device_error_.offset_bytes,
              sizeof(error));
  if (!qwen_cuda_invariant_known(error)) {
    return Status::Internal("packed result contains unknown device error");
  }
  if (error != static_cast<std::uint32_t>(QwenCudaInvariant::kNone)) {
    return Status::Internal("packed device invariant rejected publication");
  }
  std::vector<std::uint32_t> tokens(sample_count);
  if (sample_count != 0)
    std::memcpy(tokens.data(), backing.data() + sampled_token_ids_.offset_bytes,
                static_cast<std::size_t>(sample_count) * sizeof(std::uint32_t));
  for (const auto token : tokens) {
    if (token >= kVocabularySize) {
      return Status::Internal("packed sampled token is outside vocabulary");
    }
  }
  return tokens;
}

}  // namespace pih
