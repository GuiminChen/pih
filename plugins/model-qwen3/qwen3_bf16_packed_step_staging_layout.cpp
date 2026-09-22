#include "pih/model/qwen3_bf16_packed_step_staging_layout.h"

#include <cstring>
#include <cmath>
#include <limits>
#include <vector>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<QwenBf16ArenaSpan> append_span(std::uint64_t& cursor,
                                      std::uint64_t count,
                                      std::uint64_t element_bytes) {
  auto offset = checked_align_up_u64(
      cursor, QwenBf16PackedStepStagingLayout::kAlignment);
  if (!offset.ok()) return offset.status();
  auto bytes = checked_mul_u64(count, element_bytes);
  if (!bytes.ok()) return bytes.status();
  auto end = checked_add_u64(*offset, *bytes);
  if (!end.ok()) return end.status();
  cursor = *end;
  return QwenBf16ArenaSpan{*offset, *bytes};
}

template <typename T>
void copy_span(std::span<std::byte> destination, QwenBf16ArenaSpan target,
               std::span<const T> source) {
  std::memcpy(destination.data() + target.offset_bytes, source.data(),
              source.size_bytes());
}

template <typename T>
void write_scalar(std::span<std::byte> destination, QwenBf16ArenaSpan target,
                  T value) {
  std::memcpy(destination.data() + target.offset_bytes, &value, sizeof(value));
}

}  // namespace

Result<QwenBf16PackedStepStagingLayout>
QwenBf16PackedStepStagingLayout::Create(
    PackedTokenMetadataView token_metadata,
    QwenBf16PackedKvMetadataView kv_metadata) {
  if (kv_metadata.key_token_counts.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      kv_metadata.visible_handles.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      token_metadata.input_token_ids.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("packed staging shape exceeds u32");
  }
  return CreateBounded(
      static_cast<std::uint32_t>(token_metadata.input_token_ids.size()),
      static_cast<std::uint32_t>(kv_metadata.key_token_counts.size()),
      static_cast<std::uint32_t>(kv_metadata.visible_handles.size()));
}

Result<QwenBf16PackedStepStagingLayout>
QwenBf16PackedStepStagingLayout::CreateBounded(
    std::uint32_t execution_bucket_tokens, std::uint32_t sequence_count,
    std::uint32_t visible_handle_count) {
  if (execution_bucket_tokens == 0 || execution_bucket_tokens > 4096 ||
      sequence_count == 0 || sequence_count > execution_bucket_tokens ||
      visible_handle_count == 0 ||
      visible_handle_count > QwenKvSlotPool::kMaximumSlots) {
    return Status::InvalidArgument("packed staging bounds are invalid");
  }
  QwenBf16PackedStepStagingLayout result;
  result.execution_bucket_tokens_ = execution_bucket_tokens;
  result.sequence_count_ = sequence_count;
  result.visible_handle_count_ = visible_handle_count;
  std::uint64_t cursor = 0;
  const std::array<std::uint64_t, kCopySpanCount> counts{
      execution_bucket_tokens, execution_bucket_tokens,
      execution_bucket_tokens, static_cast<std::uint64_t>(sequence_count) + 1,
      sequence_count, 1, execution_bucket_tokens, execution_bucket_tokens,
      static_cast<std::uint64_t>(sequence_count) + 1, visible_handle_count,
      sequence_count, sequence_count, sequence_count, sequence_count};
  const std::array<std::uint64_t, kCopySpanCount> element_bytes{
      sizeof(std::uint32_t), sizeof(std::uint64_t), sizeof(std::uint32_t),
      sizeof(std::uint32_t), sizeof(std::uint32_t), sizeof(std::uint32_t),
      sizeof(QwenKvBlockHandle), sizeof(std::uint16_t), sizeof(std::uint32_t),
      sizeof(QwenKvBlockHandle), sizeof(std::uint32_t), sizeof(std::uint32_t),
      sizeof(Qwen3PackedSamplingWire), sizeof(std::uint32_t)};
  static_assert(kCopySpanCount == 14);
  for (std::size_t i = 0; i < result.spans_.size(); ++i) {
    auto span = append_span(cursor, counts[i], element_bytes[i]);
    if (!span.ok()) return span.status();
    result.spans_[i] = *span;
  }
  auto total = checked_align_up_u64(cursor, kAlignment);
  if (!total.ok()) return total.status();
  result.total_bytes_ = *total;
  return result;
}

Status QwenBf16PackedStepStagingLayout::materialize(
    PackedTokenMetadataView token_metadata,
    QwenBf16PackedKvMetadataView kv_metadata,
    std::span<std::byte> backing) const {
  std::vector<Qwen3SamplingDescriptor> greedy(sequence_count_);
  return materialize(token_metadata, kv_metadata, greedy, backing);
}

Status QwenBf16PackedStepStagingLayout::materialize(
    PackedTokenMetadataView token_metadata,
    QwenBf16PackedKvMetadataView kv_metadata,
    std::span<const Qwen3SamplingDescriptor> sampling,
    std::span<std::byte> backing) const {
  if (token_metadata.input_token_ids.size() != execution_bucket_tokens_ ||
      token_metadata.positions.size() != execution_bucket_tokens_ ||
      token_metadata.request_index.size() != execution_bucket_tokens_ ||
      token_metadata.query_start_offsets.size() != sequence_count_ + 1 ||
      token_metadata.sample_row_index.size() > sequence_count_ ||
      token_metadata.real_token_count > execution_bucket_tokens_ ||
      kv_metadata.append_handles.size() != execution_bucket_tokens_ ||
      kv_metadata.token_offsets.size() != execution_bucket_tokens_ ||
      kv_metadata.visible_handle_offsets.size() != sequence_count_ + 1 ||
      kv_metadata.visible_handles.size() != visible_handle_count_ ||
      kv_metadata.key_token_counts.size() != sequence_count_ ||
      kv_metadata.owner_sequence_indices.size() != sequence_count_ ||
      sampling.size() != sequence_count_ ||
      token_metadata.query_start_offsets.back() !=
          token_metadata.real_token_count ||
      kv_metadata.visible_handle_offsets.back() != visible_handle_count_ ||
      backing.size() < total_bytes_) {
    return Status::InvalidArgument("packed staging input shape drifted");
  }
  for (const auto row : token_metadata.sample_row_index) {
    if (row >= token_metadata.real_token_count) {
      return Status::InvalidArgument("packed staging sample row is invalid");
    }
  }
  for (std::size_t i = 0; i < sampling.size(); ++i) {
    const auto& descriptor = sampling[i];
    if ((descriptor.mode != Qwen3SamplingMode::kGreedy &&
         descriptor.mode != Qwen3SamplingMode::kStochastic) ||
        !std::isfinite(descriptor.temperature) ||
        !std::isfinite(descriptor.top_p) || descriptor.top_p <= 0.0F ||
        descriptor.top_p > 1.0F ||
        (descriptor.top_k.has_value() &&
         (*descriptor.top_k == 0U || *descriptor.top_k > 151936U)) ||
        descriptor.top_logprobs_count > 20U ||
        descriptor.suppressed_token_count >
            Qwen3SamplingDescriptor::kMaximumSuppressedTokenIds ||
        (descriptor.mode == Qwen3SamplingMode::kGreedy &&
         (descriptor.temperature != 0.0F || descriptor.top_p != 1.0F ||
          descriptor.top_k.has_value())) ||
        (descriptor.mode == Qwen3SamplingMode::kStochastic &&
         (descriptor.temperature <= 0.0F || descriptor.temperature > 2.0F))) {
      return Status::InvalidArgument(
          "packed staging sampling descriptor is invalid");
    }
    for (std::uint32_t token = 0;
         token < descriptor.suppressed_token_ids.size(); ++token) {
      if ((token < descriptor.suppressed_token_count &&
           descriptor.suppressed_token_ids[token] >= 151936U) ||
          (token >= descriptor.suppressed_token_count &&
           descriptor.suppressed_token_ids[token] != 0U)) {
        return Status::InvalidArgument(
            "packed staging suppressed-token descriptor is invalid");
      }
      for (std::uint32_t previous = 0;
           token < descriptor.suppressed_token_count && previous < token;
           ++previous) {
        if (descriptor.suppressed_token_ids[previous] ==
            descriptor.suppressed_token_ids[token]) {
          return Status::InvalidArgument(
              "packed staging suppressed-token set is not unique");
        }
      }
    }
  }

  copy_span(backing, token_ids(), token_metadata.input_token_ids);
  copy_span(backing, positions(), token_metadata.positions);
  copy_span(backing, request_index(), token_metadata.request_index);
  copy_span(backing, query_start_offsets(),
            token_metadata.query_start_offsets);
  for (std::uint32_t i = 0; i < sequence_count_; ++i) {
    const std::uint32_t row = i < token_metadata.sample_row_index.size()
                                  ? token_metadata.sample_row_index[i]
                                  : kInvalidPackedRequestIndex;
    std::memcpy(backing.data() + sample_row_index().offset_bytes +
                    i * sizeof(std::uint32_t),
                &row, sizeof(row));
  }
  for (std::uint32_t sample = 0; sample < sequence_count_; ++sample) {
    std::uint32_t sequence = kInvalidPackedRequestIndex;
    if (sample < token_metadata.sample_row_index.size()) {
      const auto row = token_metadata.sample_row_index[sample];
      for (std::uint32_t candidate = 0; candidate < sequence_count_;
           ++candidate) {
        if (token_metadata.query_start_offsets[candidate + 1U] != 0U &&
            token_metadata.query_start_offsets[candidate + 1U] - 1U == row) {
          sequence = candidate;
          break;
        }
      }
      if (sequence == kInvalidPackedRequestIndex) {
        return Status::InvalidArgument(
            "packed staging sample sequence mapping is invalid");
      }
    }
    std::memcpy(backing.data() + sample_sequence_indices().offset_bytes +
                    sample * sizeof(std::uint32_t),
                &sequence, sizeof(sequence));
  }
  write_scalar(backing, sample_count(), static_cast<std::uint32_t>(
      token_metadata.sample_row_index.size()));
  copy_span(backing, append_handles(), kv_metadata.append_handles);
  copy_span(backing, token_offsets(), kv_metadata.token_offsets);
  copy_span(backing, visible_handle_offsets(),
            kv_metadata.visible_handle_offsets);
  copy_span(backing, visible_handles(), kv_metadata.visible_handles);
  copy_span(backing, key_token_counts(), kv_metadata.key_token_counts);
  copy_span(backing, owner_sequence_indices(),
            kv_metadata.owner_sequence_indices);
  for (std::size_t i = 0; i < sampling.size(); ++i) {
    const auto& descriptor = sampling[i];
    const Qwen3PackedSamplingWire wire{
        descriptor.mode == Qwen3SamplingMode::kStochastic ? 1U : 0U,
        descriptor.temperature, descriptor.top_p,
        descriptor.top_k.value_or(0U), descriptor.seed,
        descriptor.sample_ordinal, descriptor.top_logprobs_count,
        descriptor.suppressed_token_ids,
        descriptor.suppressed_token_count};
    std::memcpy(backing.data() + sampling_descriptors().offset_bytes +
                    i * sizeof(Qwen3PackedSamplingWire),
                &wire, sizeof(wire));
  }
  return Status::Ok();
}

}  // namespace pih
