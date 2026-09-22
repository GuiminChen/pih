#include "pih/model/qwen3_kv_block_table.h"

#include <limits>
#include <unordered_set>

namespace pih {

Result<std::uint32_t> QwenKvBlockTable::handles_for_tokens(
    std::uint32_t tokens) {
  if (tokens == 0) return UINT32_C(0);
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(tokens) + QwenKvSlotPool::kTokensPerSlot - 1) /
      QwenKvSlotPool::kTokensPerSlot);
}

Result<QwenKvBlockTable> QwenKvBlockTable::Create(
    std::uint32_t owner_sequence_index, std::uint32_t sequence_generation,
    std::uint32_t reserved_tokens,
    std::span<const QwenKvBlockHandle> reserved_handles) {
  if (owner_sequence_index == QwenKvSlotPool::kNoOwner ||
      sequence_generation == 0 || reserved_tokens == 0 ||
      reserved_tokens > kMaximumReservedTokens) {
    return Status::InvalidArgument("Qwen KV sequence identity is invalid");
  }
  auto required = handles_for_tokens(reserved_tokens);
  if (!required.ok()) return required.status();
  if (reserved_handles.size() != required.value()) {
    return Status::InvalidArgument(
        "Qwen KV reserved handle count differs from token reservation");
  }
  std::unordered_set<std::uint32_t> slots;
  slots.reserve(reserved_handles.size());
  for (const auto& handle : reserved_handles) {
    if (handle.generation == 0 || !slots.insert(handle.slot).second) {
      return Status::InvalidArgument("Qwen KV reserved handle is invalid");
    }
  }
  QwenKvSequenceDescriptor descriptor{};
  descriptor.sequence_generation = sequence_generation;
  descriptor.owner_sequence_index = owner_sequence_index;
  descriptor.reserved_tokens = reserved_tokens;
  descriptor.committed_tokens = 0;
  descriptor.handle_count = required.value();
  descriptor.block_table_generation = 1;
  descriptor.active = 1;
  return QwenKvBlockTable(
      descriptor,
      std::vector<QwenKvBlockHandle>(reserved_handles.begin(),
                                     reserved_handles.end()));
}

Result<QwenKvAppendPlan> QwenKvBlockTable::prepare_append(
    std::uint32_t target_committed_tokens) const {
  if (descriptor_.active != 1 ||
      target_committed_tokens <= descriptor_.committed_tokens ||
      target_committed_tokens > descriptor_.reserved_tokens) {
    return Status::InvalidArgument("Qwen KV append target is outside reservation");
  }
  if (descriptor_.block_table_generation ==
      std::numeric_limits<std::uint32_t>::max()) {
    return Status::FailedPrecondition(
        "Qwen KV block-table generation requires epoch rebuild");
  }
  auto visible = handles_for_tokens(target_committed_tokens);
  if (!visible.ok()) return visible.status();
  return QwenKvAppendPlan{descriptor_.owner_sequence_index,
                          descriptor_.sequence_generation,
                          descriptor_.block_table_generation,
                          descriptor_.committed_tokens,
                          target_committed_tokens, visible.value()};
}

Status QwenKvBlockTable::validate_plan(const QwenKvAppendPlan& plan) const {
  if (descriptor_.active != 1 ||
      plan.owner_sequence_index != descriptor_.owner_sequence_index ||
      plan.expected_sequence_generation != descriptor_.sequence_generation ||
      plan.expected_block_table_generation !=
          descriptor_.block_table_generation ||
      plan.previous_committed_tokens != descriptor_.committed_tokens ||
      plan.target_committed_tokens <= descriptor_.committed_tokens ||
      plan.target_committed_tokens > descriptor_.reserved_tokens) {
    return Status::FailedPrecondition("Qwen KV append plan is stale");
  }
  auto visible = handles_for_tokens(plan.target_committed_tokens);
  if (!visible.ok()) return visible.status();
  if (visible.value() != plan.visible_handle_count_after_commit ||
      visible.value() > handles_.size()) {
    return Status::InvalidArgument("Qwen KV append plan handle count is invalid");
  }
  return Status::Ok();
}

Status QwenKvBlockTable::commit_append(const QwenKvAppendPlan& plan) {
  const Status valid = validate_plan(plan);
  if (!valid.ok()) return valid;
  if (descriptor_.block_table_generation ==
      std::numeric_limits<std::uint32_t>::max()) {
    return Status::FailedPrecondition(
        "Qwen KV block-table generation requires epoch rebuild");
  }
  descriptor_.committed_tokens = plan.target_committed_tokens;
  ++descriptor_.block_table_generation;
  return Status::Ok();
}

Status QwenKvBlockTable::rollback_append(const QwenKvAppendPlan& plan) const {
  return validate_plan(plan);
}

std::span<const QwenKvBlockHandle> QwenKvBlockTable::visible_handles() const {
  const auto count = handles_for_tokens(descriptor_.committed_tokens).value();
  return {handles_.data(), count};
}

}  // namespace pih
