#pragma once

#include <cstdint>

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_bf16_request_runner.h"

namespace pih {

class QwenBf16KvScrubDriver {
 public:
  virtual ~QwenBf16KvScrubDriver() = default;
  virtual Status clear_and_wait(std::uintptr_t destination,
                                std::uint64_t bytes,
                                DriverStreamHandle stream,
                                DriverEventHandle event) = 0;
};

class QwenBf16SynchronousKvRecycler final : public QwenBf16KvRecycler {
 public:
  static Result<QwenBf16SynchronousKvRecycler> Create(
      std::uintptr_t kv_backing, std::uint64_t kv_backing_bytes,
      std::uint32_t slot_count, DriverStreamHandle scrub_stream,
      DriverEventHandle scrub_event, std::uint64_t first_scrub_generation,
      QwenBf16KvScrubDriver& driver);

  Status recycle(QwenKvSlotPool& pool,
                 std::span<const QwenKvBlockHandle> handles,
                 QwenKvCompletionEvent last_use_event) override;

 private:
  QwenBf16SynchronousKvRecycler(
      std::uintptr_t kv_backing, std::uint64_t kv_backing_bytes,
      std::uint32_t slot_count, DriverStreamHandle scrub_stream,
      DriverEventHandle scrub_event, std::uint64_t first_scrub_generation,
      QwenBf16KvScrubDriver& driver)
      : kv_backing_(kv_backing), kv_backing_bytes_(kv_backing_bytes),
        slot_count_(slot_count), scrub_stream_(scrub_stream),
        scrub_event_(scrub_event),
        next_scrub_generation_(first_scrub_generation), driver_(&driver) {}

  std::uintptr_t kv_backing_;
  std::uint64_t kv_backing_bytes_;
  std::uint32_t slot_count_;
  DriverStreamHandle scrub_stream_;
  DriverEventHandle scrub_event_;
  std::uint64_t next_scrub_generation_;
  QwenBf16KvScrubDriver* driver_;
};

}  // namespace pih
