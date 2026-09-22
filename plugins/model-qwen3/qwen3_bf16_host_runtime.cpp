#include "pih/model/qwen3_bf16_host_runtime.h"

#include <chrono>
#include <thread>

namespace pih {

Result<std::uint64_t> QwenBf16SteadyClock::now_ns() {
  const auto elapsed = std::chrono::steady_clock::now().time_since_epoch();
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  if (nanoseconds < 0) {
    return Status::Internal("steady clock returned a negative duration");
  }
  return static_cast<std::uint64_t>(nanoseconds);
}

Status QwenBf16YieldWaiter::wait() {
  std::this_thread::yield();
  return Status::Ok();
}

}  // namespace pih
