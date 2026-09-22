#include "pih/model/qwen3_kv_sequence_lease.h"

#include <utility>

namespace pih {

QwenKvSequenceLease::QwenKvSequenceLease(
    QwenKvSequenceLease&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      table_(std::move(other.table_)),
      released_(std::exchange(other.released_, true)) {}

Result<QwenKvSequenceLease> QwenKvSequenceLease::Admit(
    QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
    std::uint32_t sequence_generation, std::uint32_t reserved_tokens) {
  if (reserved_tokens == 0 ||
      reserved_tokens > QwenKvBlockTable::kMaximumReservedTokens) {
    return Status::InvalidArgument("Qwen KV lease token reservation is invalid");
  }
  const auto slot_count = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(reserved_tokens) +
       QwenKvSlotPool::kTokensPerSlot - 1) /
      QwenKvSlotPool::kTokensPerSlot);
  auto handles = pool.reserve(owner_sequence_index, slot_count);
  if (!handles.ok()) return handles.status();
  auto table = QwenKvBlockTable::Create(owner_sequence_index,
                                        sequence_generation, reserved_tokens,
                                        handles.value());
  if (!table.ok()) {
    const Status rolled_back = pool.rollback(owner_sequence_index, handles.value());
    if (!rolled_back.ok()) {
      return Status::Internal("Qwen KV admission rollback failed");
    }
    return table.status();
  }
  const Status published = pool.publish(owner_sequence_index, handles.value());
  if (!published.ok()) {
    const Status rolled_back = pool.rollback(owner_sequence_index, handles.value());
    if (!rolled_back.ok()) {
      return Status::Internal("Qwen KV publish rollback failed");
    }
    return published;
  }
  return QwenKvSequenceLease(pool, std::move(table).value());
}

Result<QwenKvAppendPlan> QwenKvSequenceLease::prepare_append(
    std::uint32_t target_committed_tokens) const {
  if (released_) return Status::FailedPrecondition("Qwen KV lease is released");
  return table_.prepare_append(target_committed_tokens);
}

Status QwenKvSequenceLease::commit_append(const QwenKvAppendPlan& plan) {
  if (released_) return Status::FailedPrecondition("Qwen KV lease is released");
  const Status valid = table_.rollback_append(plan);
  if (!valid.ok()) return valid;
  const Status prefix = pool_->commit_prefix(
      table_.descriptor().owner_sequence_index, table_.reserved_handles(),
      plan.target_committed_tokens);
  if (!prefix.ok()) return prefix;
  return table_.commit_append(plan);
}

Status QwenKvSequenceLease::rollback_append(
    const QwenKvAppendPlan& plan) const {
  if (released_) return Status::FailedPrecondition("Qwen KV lease is released");
  return table_.rollback_append(plan);
}

Status QwenKvSequenceLease::release(
    QwenKvCompletionEvent last_use_event) {
  if (released_) return Status::FailedPrecondition("Qwen KV lease is released");
  const Status released = pool_->release(
      table_.descriptor().owner_sequence_index, table_.reserved_handles(),
      last_use_event);
  if (!released.ok()) return released;
  released_ = true;
  return Status::Ok();
}

}  // namespace pih
