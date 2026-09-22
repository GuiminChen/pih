#include "pih/model/qwen3_bf16_kv_recycler.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenBf16SynchronousKvRecycler>
QwenBf16SynchronousKvRecycler::Create(
    std::uintptr_t kv_backing, std::uint64_t kv_backing_bytes,
    std::uint32_t slot_count, DriverStreamHandle scrub_stream,
    DriverEventHandle scrub_event, std::uint64_t first_scrub_generation,
    QwenBf16KvScrubDriver& driver) {
  auto required = checked_mul_u64(slot_count,
                                  QwenKvSlotPool::kSlotPayloadBytes);
  if (!required.ok()) return required.status();
  if (kv_backing == 0 || slot_count == 0 ||
      slot_count > QwenKvSlotPool::kMaximumSlots ||
      kv_backing_bytes != *required || scrub_stream == 0 ||
      scrub_event == 0 || first_scrub_generation == 0) {
    return Status::InvalidArgument("Qwen KV recycler identity is invalid");
  }
  return QwenBf16SynchronousKvRecycler(
      kv_backing, kv_backing_bytes, slot_count, scrub_stream, scrub_event,
      first_scrub_generation, driver);
}

Status QwenBf16SynchronousKvRecycler::recycle(
    QwenKvSlotPool& pool, std::span<const QwenKvBlockHandle> handles,
    QwenKvCompletionEvent last_use_event) {
  if (!pool.ready() || pool.failed() || pool.slot_count() != slot_count_ ||
      handles.empty() || last_use_event.handle == 0 ||
      last_use_event.generation == 0) {
    return Status::FailedPrecondition("Qwen KV recycler cannot run");
  }
  for (const auto handle : handles) {
    if (handle.slot >= slot_count_ ||
        next_scrub_generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return Status::ResourceExhausted(
          "Qwen KV scrub identity or slot range is exhausted");
    }
    Status status = pool.complete_reclaim_event(
        handle, last_use_event, true, true);
    if (!status.ok()) return status;
    const QwenKvCompletionEvent scrub_event{
        scrub_event_, next_scrub_generation_++};
    auto work = pool.begin_next_scrub(scrub_event);
    if (!work.ok()) return work.status();
    if (work->bytes != QwenKvSlotPool::kSlotPayloadBytes ||
        work->slot.slot >= slot_count_) {
      return Status::FailedPrecondition(
          "Qwen KV scrub queue returned an unexpected slot");
    }
    auto offset = checked_mul_u64(work->slot.slot,
                                  QwenKvSlotPool::kSlotPayloadBytes);
    if (!offset.ok() || *offset > kv_backing_bytes_ - work->bytes ||
        *offset > std::numeric_limits<std::uintptr_t>::max() - kv_backing_) {
      return Status::Internal("Qwen KV scrub address exceeds its backing");
    }
    status = driver_->clear_and_wait(
        kv_backing_ + static_cast<std::uintptr_t>(*offset), work->bytes,
        scrub_stream_, scrub_event_);
    const Status completed = pool.complete_scrub(
        work->slot, scrub_event, status.ok() ? work->bytes : 0, true,
        status.ok());
    if (!status.ok()) return status;
    if (!completed.ok()) return completed;
  }
  return Status::Ok();
}

}  // namespace pih
