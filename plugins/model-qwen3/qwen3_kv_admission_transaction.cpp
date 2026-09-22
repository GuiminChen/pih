#include "pih/model/qwen3_kv_admission_transaction.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenKvAdmissionTransaction> QwenKvAdmissionTransaction::Reserve(
    QwenKvSlotPool& pool, std::uint32_t owner_sequence_index,
    std::uint32_t reserved_tokens) {
  if (owner_sequence_index == QwenKvSlotPool::kNoOwner ||
      reserved_tokens == 0 ||
      reserved_tokens > QwenKvBlockTable::kMaximumReservedTokens) {
    return Status::InvalidArgument("Qwen KV admission bounds are invalid");
  }
  auto rounded = checked_add_u64(
      reserved_tokens, QwenKvSlotPool::kTokensPerSlot - 1);
  if (!rounded.ok()) return rounded.status();
  const auto slot_count = static_cast<std::uint32_t>(
      *rounded / QwenKvSlotPool::kTokensPerSlot);
  auto sequence_generation = pool.acquire_sequence_generation();
  if (!sequence_generation.ok()) return sequence_generation.status();
  auto handles = pool.reserve(owner_sequence_index, slot_count);
  if (!handles.ok()) return handles.status();
  return QwenKvAdmissionTransaction(
      pool, owner_sequence_index, *sequence_generation, reserved_tokens,
      std::move(*handles));
}

Status QwenKvAdmissionTransaction::publish() {
  if (state_ != QwenKvAdmissionState::kReserved) {
    return Status::FailedPrecondition(
        "Qwen KV admission is not reserved for publication");
  }
  auto table = QwenKvBlockTable::Create(
      owner_sequence_index_, sequence_generation_, reserved_tokens_, handles_);
  if (!table.ok()) {
    const Status restored = pool_->rollback(owner_sequence_index_, handles_);
    state_ = restored.ok() ? QwenKvAdmissionState::kRolledBack
                           : QwenKvAdmissionState::kFailed;
    return table.status();
  }
  const Status published = pool_->publish(owner_sequence_index_, handles_);
  if (!published.ok()) {
    const Status restored = pool_->rollback(owner_sequence_index_, handles_);
    state_ = restored.ok() ? QwenKvAdmissionState::kRolledBack
                           : QwenKvAdmissionState::kFailed;
    return published;
  }
  table_.emplace(std::move(*table));
  state_ = QwenKvAdmissionState::kPublished;
  return Status::Ok();
}

Status QwenKvAdmissionTransaction::rollback() {
  if (state_ == QwenKvAdmissionState::kRolledBack) return Status::Ok();
  if (state_ != QwenKvAdmissionState::kReserved) {
    return Status::FailedPrecondition(
        "published Qwen KV admission cannot be rolled back");
  }
  const Status restored = pool_->rollback(owner_sequence_index_, handles_);
  state_ = restored.ok() ? QwenKvAdmissionState::kRolledBack
                         : QwenKvAdmissionState::kFailed;
  return restored;
}

}  // namespace pih
