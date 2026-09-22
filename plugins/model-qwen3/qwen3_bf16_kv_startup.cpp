#include "pih/model/qwen3_bf16_kv_startup.h"

#include "pih/core/checked_math.h"

namespace pih {

Status QwenBf16KvStartup::SanitizeAndPublish(
    QwenKvSlotPool& pool, const QwenBf16DeviceArenaOwner& kv_backing,
    const QwenBf16DeviceArenaOwner& kv_metadata,
    DriverStreamHandle scrub_stream, DriverEventHandle scrub_event,
    QwenBf16KvScrubDriver& driver) {
  if (pool.ready() || pool.failed() || scrub_stream == 0 || scrub_event == 0 ||
      kv_backing.base == 0 || kv_backing.generation == 0 ||
      kv_metadata.base == 0 || kv_metadata.generation == 0) {
    return Status::FailedPrecondition(
        "Qwen KV startup sanitizer dependencies are invalid");
  }
  auto expected_kv = checked_mul_u64(pool.slot_count(),
                                     QwenKvSlotPool::kSlotPayloadBytes);
  auto expected_metadata = checked_mul_u64(pool.slot_count(),
                                           sizeof(QwenKvSlotState));
  if (!expected_kv.ok()) return expected_kv.status();
  if (!expected_metadata.ok()) return expected_metadata.status();
  if (kv_backing.bytes != *expected_kv ||
      kv_metadata.bytes != *expected_metadata) {
    return Status::InvalidArgument(
        "Qwen KV startup backing does not match the host pool");
  }
  Status status = driver.clear_and_wait(
      kv_backing.base, kv_backing.bytes, scrub_stream, scrub_event);
  if (!status.ok()) {
    (void)pool.complete_startup_sanitize(0, 0, false);
    return status;
  }
  status = driver.clear_and_wait(
      kv_metadata.base, kv_metadata.bytes, scrub_stream, scrub_event);
  if (!status.ok()) {
    (void)pool.complete_startup_sanitize(kv_backing.bytes, 0, false);
    return status;
  }
  return pool.complete_startup_sanitize(
      kv_backing.bytes, kv_metadata.bytes, true);
}

}  // namespace pih
