#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct EngramCompletionResources final {
  // Four bytes of pinned host memory and an exclusive CUDA event. Both remain
  // owned by the caller and must not be reused until completion or safe abort.
  EngramDeviceRegion host_error_flag;
  std::uintptr_t event = 0;
};
Status ValidateEngramCompletionResources(const EngramCompletionResources& resources);
class EngramCompletion final {
 public:
  EngramCompletion(const EngramCompletion&) = delete;
  EngramCompletion& operator=(const EngramCompletion&) = delete;
  EngramCompletion(EngramCompletion&& other) noexcept;
  EngramCompletion& operator=(EngramCompletion&&) = delete;
  // Records error readback and event after the gate on its stream. Failure may
  // leave queued work; caller retains resources until safely retired.
  static Result<EngramCompletion> Record(const EngramGateLaunch& gate,
      const EngramCompletionResources& resources);
  static Result<EngramCompletion> RecordFlag(EngramDeviceRegion error_flag,
      std::uintptr_t stream, const EngramCompletionResources& resources);
  // false = pending, true = event completed AND zero device error flag.
  Result<bool> Poll();
 private:
  EngramCompletion() = default;
  EngramCompletionResources resources_{};
  bool failed_ = true, complete_ = false;
};
}  // namespace pih::deepseek_v41
