#pragma once

#include "pih/model/qwen3_bf16_synchronous_backend.h"
#include "pih/model/qwen3_int4_weight_startup.h"

namespace pih {

class QwenBf16SteadyClock final : public QwenBf16MonotonicClock,
                                  public QwenInt4StartupClock {
 public:
  Result<std::uint64_t> now_ns() override;
};

class QwenBf16YieldWaiter final : public QwenBf16PollWaiter,
                                  public QwenInt4StartupWaiter {
 public:
  Status wait() override;
};

}  // namespace pih
