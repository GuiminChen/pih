#include "pih/model/qwen3_bf16_packed_kv_metadata.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenBf16PackedKvMetadataArena>
QwenBf16PackedKvMetadataArena::Create(
    QwenBf16PackedKvMetadataLimits limits) {
  auto maximum_possible_handles = checked_mul_u64(
      limits.maximum_sequences,
      (QwenKvBlockTable::kMaximumReservedTokens +
       QwenKvSlotPool::kTokensPerSlot - 1) /
          QwenKvSlotPool::kTokensPerSlot);
  if (limits.maximum_sequences == 0 ||
      limits.maximum_sequences > 4096 ||
      limits.maximum_execution_bucket_tokens == 0 ||
      limits.maximum_execution_bucket_tokens > 4096 ||
      limits.maximum_visible_handles < limits.maximum_sequences ||
      !maximum_possible_handles.ok() ||
      limits.maximum_visible_handles > *maximum_possible_handles) {
    return Status::InvalidArgument("packed KV metadata limits are invalid");
  }
  return QwenBf16PackedKvMetadataArena(limits);
}

Result<QwenBf16PackedKvMetadataView>
QwenBf16PackedKvMetadataArena::materialize(
    const PackedTokenPlan& plan, PackedTokenMetadataView token_metadata,
    std::span<const QwenBf16PackedKvBinding> bindings,
    std::span<const QwenKvAppendPlan> append_plans) {
  const auto sequence_count = plan.sequence_count();
  if (sequence_count == 0 || sequence_count > limits_.maximum_sequences ||
      bindings.size() != sequence_count ||
      append_plans.size() != sequence_count ||
      plan.execution_bucket_tokens() >
          limits_.maximum_execution_bucket_tokens ||
      token_metadata.input_token_ids.size() !=
          plan.execution_bucket_tokens() ||
      token_metadata.positions.size() != plan.execution_bucket_tokens() ||
      token_metadata.request_index.size() !=
          plan.execution_bucket_tokens() ||
      token_metadata.real_token_count != plan.total_real_tokens() ||
      generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return Status::InvalidArgument("packed KV metadata shape is invalid");
  }

  std::uint64_t visible_count = 0;
  for (std::size_t i = 0; i < sequence_count; ++i) {
    const auto& binding = bindings[i];
    const auto& append = append_plans[i];
    if (binding.block_table == nullptr ||
        binding.sequence_generation !=
            plan.ordered_sequence_generations()[i] ||
        binding.block_table->descriptor().active != 1 ||
        binding.block_table->descriptor().committed_tokens !=
            plan.committed_start_positions()[i] ||
        append.previous_committed_tokens !=
            plan.committed_start_positions()[i]) {
      return Status::FailedPrecondition("packed KV binding cut drifted");
    }
    auto target = checked_add_u64(plan.committed_start_positions()[i],
                                  plan.real_token_counts()[i]);
    if (!target.ok() || *target != append.target_committed_tokens) {
      return Status::InvalidArgument("packed KV append target drifted");
    }
    const auto valid = binding.block_table->rollback_append(append);
    if (!valid.ok()) return valid;
    if (append.visible_handle_count_after_commit >
        binding.block_table->reserved_handles().size()) {
      return Status::InvalidArgument("packed KV visible table overflowed");
    }
    auto next_visible = checked_add_u64(
        visible_count, append.visible_handle_count_after_commit);
    if (!next_visible.ok() ||
        *next_visible > limits_.maximum_visible_handles) {
      return Status::ResourceExhausted(
          "packed KV visible-handle arena is too small");
    }
    visible_count = *next_visible;

    const auto begin = plan.packed_offsets()[i];
    const auto end = plan.packed_offsets()[i + 1];
    for (std::uint32_t packed = begin; packed < end; ++packed) {
      const auto position = token_metadata.positions[packed];
      const auto handle_index =
          position / QwenKvSlotPool::kTokensPerSlot;
      if (token_metadata.request_index[packed] != i ||
          position != plan.committed_start_positions()[i] + packed - begin ||
          handle_index >= binding.block_table->reserved_handles().size()) {
        return Status::InvalidArgument("packed KV row mapping drifted");
      }
    }
  }
  for (std::uint32_t packed = plan.total_real_tokens();
       packed < plan.execution_bucket_tokens(); ++packed) {
    if (token_metadata.positions[packed] != 0 ||
        token_metadata.request_index[packed] != kInvalidPackedRequestIndex) {
      return Status::InvalidArgument("packed KV dummy row drifted");
    }
  }

  std::uint32_t visible_offset = 0;
  for (std::size_t i = 0; i < sequence_count; ++i) {
    const auto& table = *bindings[i].block_table;
    const auto& append = append_plans[i];
    visible_handle_offsets_[i] = visible_offset;
    const auto handles = table.reserved_handles().first(
        append.visible_handle_count_after_commit);
    for (const auto handle : handles) {
      visible_handles_[visible_offset++] = handle;
    }
    key_token_counts_[i] = append.target_committed_tokens;
    owner_sequence_indices_[i] = table.descriptor().owner_sequence_index;
    const auto begin = plan.packed_offsets()[i];
    const auto end = plan.packed_offsets()[i + 1];
    for (std::uint32_t packed = begin; packed < end; ++packed) {
      const auto position = token_metadata.positions[packed];
      append_handles_[packed] = table.reserved_handles()[
          position / QwenKvSlotPool::kTokensPerSlot];
      token_offsets_[packed] = static_cast<std::uint16_t>(
          position % QwenKvSlotPool::kTokensPerSlot);
    }
  }
  visible_handle_offsets_[sequence_count] = visible_offset;
  for (std::uint32_t packed = plan.total_real_tokens();
       packed < plan.execution_bucket_tokens(); ++packed) {
    append_handles_[packed] = kInvalidPackedKvBlockHandle;
    token_offsets_[packed] = 0;
  }
  ++generation_;
  return QwenBf16PackedKvMetadataView{
      generation_,
      std::span(append_handles_).first(plan.execution_bucket_tokens()),
      std::span(token_offsets_).first(plan.execution_bucket_tokens()),
      std::span(visible_handle_offsets_).first(sequence_count + 1),
      std::span(visible_handles_).first(visible_offset),
      std::span(key_token_counts_).first(sequence_count),
      std::span(owner_sequence_indices_).first(sequence_count)};
}

}  // namespace pih
