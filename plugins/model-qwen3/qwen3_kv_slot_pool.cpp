#include "pih/model/qwen3_kv_slot_pool.h"

#include <limits>
#include <unordered_set>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenKvSlotPool> QwenKvSlotPool::Create(
    std::uint32_t slot_count, std::uint64_t kv_backing_bytes,
    std::uint64_t metadata_backing_bytes) {
  if (slot_count == 0 || slot_count > kMaximumSlots) {
    return Status::InvalidArgument("Qwen KV slot count is outside profile bounds");
  }
  auto required_kv = checked_mul_u64(slot_count, kSlotPayloadBytes);
  auto required_metadata = checked_mul_u64(slot_count, sizeof(QwenKvSlotState));
  if (!required_kv.ok()) return required_kv.status();
  if (!required_metadata.ok()) return required_metadata.status();
  if (kv_backing_bytes < required_kv.value() ||
      kv_backing_bytes - required_kv.value() >= kSlotPayloadBytes ||
      metadata_backing_bytes < required_metadata.value()) {
    return Status::InvalidArgument("Qwen KV backing does not match slot capacity");
  }
  std::vector<QwenKvSlotState> slots(
      slot_count, QwenKvSlotState{0, kNoOwner, 0,
                                  QwenKvSlotLifecycle::kUninitialized, 0, 0});
  std::vector<QwenKvCompletionEvent> pending_events(slot_count, {0, 0});
  return QwenKvSlotPool(std::move(slots), std::move(pending_events),
                        kv_backing_bytes,
                        metadata_backing_bytes);
}

Status QwenKvSlotPool::complete_startup_sanitize(
    std::uint64_t cleared_kv_bytes, std::uint64_t cleared_metadata_bytes,
    bool completion_event_succeeded) {
  if (failed_ || ready_) {
    return Status::FailedPrecondition(
        "Qwen KV startup sanitize is not in initial state");
  }
  if (!completion_event_succeeded || cleared_kv_bytes != kv_backing_bytes_ ||
      cleared_metadata_bytes != metadata_backing_bytes_) {
    failed_ = true;
    return Status::FailedPrecondition(
        "Qwen KV startup sanitize did not cover the complete backing");
  }
  for (auto& slot : slots_) {
    slot = {1, kNoOwner, 0, QwenKvSlotLifecycle::kFreeClean, 0, 0};
  }
  clean_credits_ = static_cast<std::uint32_t>(slots_.size());
  ready_ = true;
  return Status::Ok();
}

Result<std::uint32_t> QwenKvSlotPool::acquire_sequence_generation() {
  if (!ready_ || failed_) {
    return Status::FailedPrecondition("Qwen KV pool is not ready");
  }
  if (next_sequence_generation_ == UINT32_MAX) {
    failed_ = true;
    return Status::ResourceExhausted(
        "Qwen sequence generation identity is exhausted");
  }
  return next_sequence_generation_++;
}

Result<std::vector<QwenKvBlockHandle>> QwenKvSlotPool::reserve(
    std::uint32_t owner_sequence_index, std::uint32_t slot_count) {
  if (!ready_ || failed_) {
    return Status::FailedPrecondition("Qwen KV pool is not ready");
  }
  if (owner_sequence_index == kNoOwner || slot_count == 0) {
    return Status::InvalidArgument("Qwen KV reservation identity is invalid");
  }
  for (const auto& slot : slots_) {
    if (slot.owner_sequence_index == owner_sequence_index) {
      return Status::FailedPrecondition(
          "Qwen KV owner already has a live reservation or allocation");
    }
  }
  if (slot_count > clean_credits_) {
    return Status::ResourceExhausted("Qwen KV clean credits are exhausted");
  }
  std::vector<std::uint32_t> selected;
  selected.reserve(slot_count);
  for (std::uint32_t ordinal = 0;
       ordinal < slots_.size() && selected.size() < slot_count; ++ordinal) {
    const auto& slot = slots_[ordinal];
    if (slot.state == QwenKvSlotLifecycle::kFreeClean) {
      if (slot.generation == std::numeric_limits<std::uint32_t>::max()) {
        return Status::FailedPrecondition(
            "Qwen KV generation wrap requires an epoch rebuild");
      }
      selected.push_back(ordinal);
    }
  }
  if (selected.size() != slot_count) {
    failed_ = true;
    return Status::Internal("Qwen KV clean credit ledger is inconsistent");
  }
  std::vector<QwenKvBlockHandle> handles;
  handles.reserve(slot_count);
  for (const auto ordinal : selected) {
    auto& slot = slots_[ordinal];
    ++slot.generation;
    slot.owner_sequence_index = owner_sequence_index;
    slot.valid_tokens = 0;
    slot.state = QwenKvSlotLifecycle::kAdmissionReserved;
    handles.push_back({ordinal, slot.generation});
  }
  clean_credits_ -= slot_count;
  return handles;
}

Status QwenKvSlotPool::validate_handle(
    const QwenKvBlockHandle& handle, std::uint32_t owner,
    QwenKvSlotLifecycle expected) const {
  if (handle.slot >= slots_.size()) {
    return Status::InvalidArgument("Qwen KV handle slot is out of range");
  }
  const auto& slot = slots_[handle.slot];
  if (handle.generation != slot.generation ||
      slot.owner_sequence_index != owner || slot.state != expected ||
      slot.reserved_zero_u8 != 0 || slot.reserved_zero_u32 != 0) {
    return Status::FailedPrecondition("Qwen KV handle is stale or in wrong state");
  }
  return Status::Ok();
}

Status QwenKvSlotPool::publish(
    std::uint32_t owner_sequence_index,
    std::span<const QwenKvBlockHandle> handles) {
  if (!ready_ || failed_ || handles.empty()) {
    return Status::FailedPrecondition("Qwen KV publish is not permitted");
  }
  const Status exact = validate_exact_owner_set(
      owner_sequence_index, handles,
      QwenKvSlotLifecycle::kAdmissionReserved);
  if (!exact.ok()) return exact;
  for (const auto& handle : handles) {
    slots_[handle.slot].state = QwenKvSlotLifecycle::kOwned;
  }
  return Status::Ok();
}

Status QwenKvSlotPool::rollback(
    std::uint32_t owner_sequence_index,
    std::span<const QwenKvBlockHandle> handles) {
  if (!ready_ || failed_ || handles.empty()) {
    return Status::FailedPrecondition("Qwen KV rollback is not permitted");
  }
  const Status exact = validate_exact_owner_set(
      owner_sequence_index, handles,
      QwenKvSlotLifecycle::kAdmissionReserved);
  if (!exact.ok()) return exact;
  for (const auto& handle : handles) {
    auto& slot = slots_[handle.slot];
    slot.owner_sequence_index = kNoOwner;
    slot.valid_tokens = 0;
    slot.state = QwenKvSlotLifecycle::kFreeClean;
  }
  clean_credits_ += static_cast<std::uint32_t>(handles.size());
  return Status::Ok();
}

Status QwenKvSlotPool::validate_exact_owner_set(
    std::uint32_t owner, std::span<const QwenKvBlockHandle> handles,
    QwenKvSlotLifecycle expected) const {
  std::size_t expected_count = 0;
  for (const auto& slot : slots_) {
    if (slot.owner_sequence_index == owner && slot.state == expected) {
      ++expected_count;
    }
  }
  if (handles.size() != expected_count) {
    return Status::FailedPrecondition(
        "Qwen KV transaction handle set is incomplete");
  }
  std::unordered_set<std::uint32_t> unique;
  unique.reserve(handles.size());
  for (const auto& handle : handles) {
    if (!unique.insert(handle.slot).second) {
      return Status::InvalidArgument("Qwen KV handle is duplicated");
    }
    const Status valid = validate_handle(handle, owner, expected);
    if (!valid.ok()) return valid;
  }
  return Status::Ok();
}

Status QwenKvSlotPool::commit_valid_tokens(
    std::uint32_t owner_sequence_index, QwenKvBlockHandle handle,
    std::uint16_t valid_tokens) {
  if (!ready_ || failed_ || valid_tokens == 0 ||
      valid_tokens > kTokensPerSlot) {
    return Status::FailedPrecondition(
        "Qwen KV valid-token commit is not permitted");
  }
  const Status valid = validate_handle(
      handle, owner_sequence_index, QwenKvSlotLifecycle::kOwned);
  if (!valid.ok()) return valid;
  auto& slot = slots_[handle.slot];
  if (valid_tokens < slot.valid_tokens) {
    return Status::InvalidArgument("Qwen KV valid tokens cannot move backward");
  }
  slot.valid_tokens = valid_tokens;
  return Status::Ok();
}

Status QwenKvSlotPool::commit_prefix(
    std::uint32_t owner_sequence_index,
    std::span<const QwenKvBlockHandle> handles,
    std::uint32_t committed_tokens) {
  if (!ready_ || failed_ || handles.empty() || committed_tokens == 0 ||
      committed_tokens >
          static_cast<std::uint64_t>(handles.size()) * kTokensPerSlot) {
    return Status::InvalidArgument("Qwen KV committed prefix is invalid");
  }
  const Status exact = validate_exact_owner_set(
      owner_sequence_index, handles, QwenKvSlotLifecycle::kOwned);
  if (!exact.ok()) return exact;
  for (std::size_t index = 0; index < handles.size(); ++index) {
    const std::uint64_t begin = index * kTokensPerSlot;
    const std::uint64_t remaining =
        committed_tokens > begin ? committed_tokens - begin : 0;
    const auto target = static_cast<std::uint16_t>(
        remaining > kTokensPerSlot ? kTokensPerSlot : remaining);
    if (target < slots_[handles[index].slot].valid_tokens) {
      return Status::InvalidArgument(
          "Qwen KV committed prefix cannot move backward");
    }
  }
  for (std::size_t index = 0; index < handles.size(); ++index) {
    const std::uint64_t begin = index * kTokensPerSlot;
    const std::uint64_t remaining =
        committed_tokens > begin ? committed_tokens - begin : 0;
    slots_[handles[index].slot].valid_tokens = static_cast<std::uint16_t>(
        remaining > kTokensPerSlot ? kTokensPerSlot : remaining);
  }
  return Status::Ok();
}

Result<std::vector<QwenKvSlotState>> QwenKvSlotPool::project_prefix(
    std::uint32_t owner_sequence_index,
    std::span<const QwenKvBlockHandle> handles,
    std::uint32_t committed_tokens) const {
  if (!ready_ || failed_ || handles.empty() || committed_tokens == 0 ||
      committed_tokens >
          static_cast<std::uint64_t>(handles.size()) * kTokensPerSlot) {
    return Status::InvalidArgument("Qwen KV projected prefix is invalid");
  }
  const Status exact = validate_exact_owner_set(
      owner_sequence_index, handles, QwenKvSlotLifecycle::kOwned);
  if (!exact.ok()) return exact;
  auto projected = slots_;
  for (std::size_t index = 0; index < handles.size(); ++index) {
    const std::uint64_t begin = index * kTokensPerSlot;
    const std::uint64_t remaining =
        committed_tokens > begin ? committed_tokens - begin : 0;
    const auto target = static_cast<std::uint16_t>(
        remaining > kTokensPerSlot ? kTokensPerSlot : remaining);
    if (target < projected[handles[index].slot].valid_tokens) {
      return Status::InvalidArgument(
          "Qwen KV projected prefix cannot move backward");
    }
    projected[handles[index].slot].valid_tokens = target;
  }
  return projected;
}

Status QwenKvSlotPool::release(
    std::uint32_t owner_sequence_index,
    std::span<const QwenKvBlockHandle> handles,
    QwenKvCompletionEvent last_use_event) {
  if (!ready_ || failed_ || handles.empty() || last_use_event.handle == 0 ||
      last_use_event.generation == 0) {
    return Status::FailedPrecondition("Qwen KV release is not permitted");
  }
  const Status exact = validate_exact_owner_set(
      owner_sequence_index, handles, QwenKvSlotLifecycle::kOwned);
  if (!exact.ok()) return exact;
  for (const auto& handle : handles) {
    slots_[handle.slot].state = QwenKvSlotLifecycle::kReclaimPending;
    pending_events_[handle.slot] = last_use_event;
  }
  return Status::Ok();
}

Status QwenKvSlotPool::complete_reclaim_event(
    QwenKvBlockHandle handle, QwenKvCompletionEvent event, bool event_ready,
    bool event_succeeded) {
  if (!ready_ || failed_ || handle.slot >= slots_.size()) {
    return Status::FailedPrecondition(
        "Qwen KV reclaim completion is not permitted");
  }
  const auto& slot = slots_[handle.slot];
  if (slot.state != QwenKvSlotLifecycle::kReclaimPending ||
      slot.generation != handle.generation ||
      pending_events_[handle.slot].handle != event.handle ||
      pending_events_[handle.slot].generation != event.generation) {
    return Status::FailedPrecondition("Qwen KV reclaim event is stale");
  }
  if (!event_ready) {
    return Status::Unavailable("Qwen KV last-use event is not complete");
  }
  if (!event_succeeded) {
    failed_ = true;
    return Status::FailedPrecondition("Qwen KV last-use event failed");
  }
  auto& mutable_slot = slots_[handle.slot];
  mutable_slot.owner_sequence_index = kNoOwner;
  mutable_slot.valid_tokens = 0;
  mutable_slot.state = QwenKvSlotLifecycle::kFreeDirty;
  pending_events_[handle.slot] = {0, 0};
  return Status::Ok();
}

Result<QwenKvScrubWork> QwenKvSlotPool::begin_next_scrub(
    QwenKvCompletionEvent scrub_completion_event) {
  if (!ready_ || failed_ || scrub_completion_event.handle == 0 ||
      scrub_completion_event.generation == 0) {
    return Status::FailedPrecondition("Qwen KV scrub cannot begin");
  }
  if (lifecycle_count(QwenKvSlotLifecycle::kScrubbing) != 0) {
    return Status::ResourceExhausted("Qwen KV scrub lane is busy");
  }
  for (std::uint32_t ordinal = 0; ordinal < slots_.size(); ++ordinal) {
    auto& slot = slots_[ordinal];
    if (slot.state == QwenKvSlotLifecycle::kFreeDirty) {
      slot.state = QwenKvSlotLifecycle::kScrubbing;
      pending_events_[ordinal] = scrub_completion_event;
      return QwenKvScrubWork{{ordinal, slot.generation}, kSlotPayloadBytes};
    }
  }
  return Status::Unavailable("Qwen KV dirty ring is empty");
}

Status QwenKvSlotPool::complete_scrub(
    QwenKvBlockHandle handle, QwenKvCompletionEvent event,
    std::uint64_t cleared_bytes, bool event_ready, bool event_succeeded) {
  if (!ready_ || failed_ || handle.slot >= slots_.size()) {
    return Status::FailedPrecondition("Qwen KV scrub completion is not permitted");
  }
  const auto& slot = slots_[handle.slot];
  if (slot.state != QwenKvSlotLifecycle::kScrubbing ||
      slot.generation != handle.generation ||
      pending_events_[handle.slot].handle != event.handle ||
      pending_events_[handle.slot].generation != event.generation) {
    return Status::FailedPrecondition("Qwen KV scrub event is stale");
  }
  if (!event_ready) return Status::Unavailable("Qwen KV scrub is not complete");
  if (!event_succeeded || cleared_bytes != kSlotPayloadBytes) {
    failed_ = true;
    return Status::FailedPrecondition(
        "Qwen KV scrub did not clear the complete slot");
  }
  auto& mutable_slot = slots_[handle.slot];
  mutable_slot.owner_sequence_index = kNoOwner;
  mutable_slot.valid_tokens = 0;
  mutable_slot.state = QwenKvSlotLifecycle::kFreeClean;
  pending_events_[handle.slot] = {0, 0};
  ++clean_credits_;
  return Status::Ok();
}

std::uint32_t QwenKvSlotPool::lifecycle_count(
    QwenKvSlotLifecycle state) const noexcept {
  std::uint32_t count = 0;
  for (const auto& slot : slots_) {
    if (slot.state == state) ++count;
  }
  return count;
}

}  // namespace pih
