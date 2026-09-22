#pragma once

#include "pih/model/qwen3_bf16_kv_recycler.h"
#include "pih/model/qwen3_bf16_step_resource_factory.h"

namespace pih {

class QwenBf16KvStartup final {
 public:
  static Status SanitizeAndPublish(
      QwenKvSlotPool& pool, const QwenBf16DeviceArenaOwner& kv_backing,
      const QwenBf16DeviceArenaOwner& kv_metadata,
      DriverStreamHandle scrub_stream, DriverEventHandle scrub_event,
      QwenBf16KvScrubDriver& driver);
};

}  // namespace pih
