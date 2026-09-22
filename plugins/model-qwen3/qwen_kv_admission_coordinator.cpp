#include "pih/scheduler/qwen_kv_admission_coordinator.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenKvAdmissionCoordinator> QwenKvAdmissionCoordinator::Create(
    QwenKvSlotPool& pool, ControllerPackedDriver& driver,
    std::uint32_t maximum_sequences,
    QwenKvCompletionEvent initial_last_use_event) {
  if (!pool.ready() || pool.failed() || maximum_sequences == 0 ||
      initial_last_use_event.handle == 0 ||
      initial_last_use_event.generation == 0) {
    return Status::InvalidArgument(
        "Qwen KV admission coordinator identity is invalid");
  }
  return QwenKvAdmissionCoordinator(
      pool, driver, maximum_sequences, initial_last_use_event);
}

std::optional<std::uint32_t> QwenKvAdmissionCoordinator::find(
    std::uint64_t sequence_generation) const noexcept {
  for (std::uint32_t index = 0; index < admissions_.size(); ++index) {
    if (admissions_[index].has_value() &&
        admissions_[index]->sequence_generation() == sequence_generation) {
      return index;
    }
  }
  return std::nullopt;
}

Status QwenKvAdmissionCoordinator::reserve(
    const ControllerAdmissionView& admission) {
  if (find(admission.request.request_generation).has_value()) {
    return Status::FailedPrecondition(
        "Qwen KV admission generation is already active");
  }
  std::uint32_t free_index = static_cast<std::uint32_t>(admissions_.size());
  for (std::uint32_t index = 0; index < admissions_.size(); ++index) {
    if (!admissions_[index].has_value()) {
      free_index = index;
      break;
    }
  }
  if (free_index == admissions_.size()) {
    return Status::ResourceExhausted(
        "Qwen KV admission sequence credits are exhausted");
  }
  auto total_tokens = checked_add_u64(
      admission.request.prompt_token_ids.size(),
      admission.request.maximum_new_tokens);
  if (!total_tokens.ok() || *total_tokens == 0 ||
      *total_tokens > QwenKvBlockTable::kMaximumReservedTokens) {
    return Status::InvalidArgument(
        "Qwen KV admission lifetime token bound is invalid");
  }
  auto transaction = QwenKvAdmissionTransaction::Reserve(
      *pool_, free_index, static_cast<std::uint32_t>(*total_tokens));
  if (!transaction.ok()) return transaction.status();
  if (transaction->sequence_generation() !=
      admission.request.request_generation) {
    const Status restored = transaction->rollback();
    if (!restored.ok()) {
      return Status::Internal(
          "Qwen KV admission generation rollback failed");
    }
    return Status::FailedPrecondition(
        "Qwen KV and controller sequence generations drifted");
  }
  admissions_[free_index].emplace(std::move(*transaction));
  return Status::Ok();
}

Status QwenKvAdmissionCoordinator::publish(
    std::uint64_t sequence_generation) {
  const auto index = find(sequence_generation);
  if (!index.has_value() ||
      admissions_[*index]->state() != QwenKvAdmissionState::kReserved) {
    return Status::FailedPrecondition(
        "Qwen KV admission publication has no reservation");
  }
  Status status = admissions_[*index]->publish();
  if (!status.ok()) return status;
  auto* table = admissions_[*index]->block_table();
  if (table == nullptr) {
    return Status::Internal("Qwen KV admission published no block table");
  }
  return driver_->bind_sequence(
      sequence_generation, *table, initial_last_use_event_);
}

Status QwenKvAdmissionCoordinator::rollback(
    std::uint64_t request_generation) {
  const auto index = find(request_generation);
  if (!index.has_value()) return Status::Ok();
  const Status status = admissions_[*index]->rollback();
  if (status.ok()) admissions_[*index].reset();
  return status;
}

Status QwenKvAdmissionCoordinator::drain(
    std::uint64_t sequence_generation, QwenBf16KvRecycler& recycler) {
  const auto index = find(sequence_generation);
  if (!index.has_value() ||
      admissions_[*index]->state() != QwenKvAdmissionState::kPublished) {
    return Status::FailedPrecondition(
        "Qwen KV drain has no published admission");
  }
  const Status status = driver_->drain_sequence(
      sequence_generation, *pool_, recycler);
  if (status.ok()) admissions_[*index].reset();
  return status;
}

std::uint32_t QwenKvAdmissionCoordinator::active_admission_count()
    const noexcept {
  std::uint32_t count = 0;
  for (const auto& admission : admissions_) {
    if (admission.has_value()) ++count;
  }
  return count;
}

}  // namespace pih
