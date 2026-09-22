#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

enum class CudaCompletionPhase : std::uint8_t {
  kPrefill = 0, kDecode, kKvAppend, kAttention, kCopy,
};
enum class CudaEventQueryResult : std::uint8_t {
  kSuccess = 0, kNotReady, kError,
};
enum class CudaFrontierFailure : std::uint8_t {
  kNone = 0,
  kEventGenerationMismatch,
  kUnexpectedQueryError,
  kLastErrorDirty,
  kDeviceInvariant,
  kEnginePoisoned,
  kDeadlineExpired,
  kInvalidObservation,
};

struct CudaErrorRecordKey final {
  std::uint64_t epoch;
  std::uint32_t rank;
  std::uint64_t plan_generation;
  CudaCompletionPhase phase;
  std::uint64_t completion_frontier;
};

class CudaCompletionFrontier final {
 public:
  static Result<CudaCompletionFrontier> Create(
      CudaErrorRecordKey key, std::uint64_t event_generation,
      std::uint64_t submit_ns, std::uint64_t deadline_ns);

  Status observe(std::uint64_t observed_event_generation,
                 CudaEventQueryResult query_result,
                 bool submit_thread_last_error_clean,
                 std::uint32_t device_error_code, bool engine_poisoned);
  Status expire(std::uint64_t now_ns);

  [[nodiscard]] bool publication_authorized() const noexcept {
    return completed_ && !poisoned_;
  }
  [[nodiscard]] bool completed() const noexcept { return completed_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] bool draining() const noexcept { return draining_; }
  [[nodiscard]] CudaFrontierFailure first_failure() const noexcept {
    return first_failure_;
  }
  [[nodiscard]] std::uint32_t first_device_error_code() const noexcept {
    return first_device_error_code_;
  }
  [[nodiscard]] const CudaErrorRecordKey& key() const noexcept { return key_; }
  [[nodiscard]] std::uint64_t event_generation() const noexcept {
    return event_generation_;
  }

 private:
  CudaCompletionFrontier(CudaErrorRecordKey key,
                         std::uint64_t event_generation,
                         std::uint64_t submit_ns, std::uint64_t deadline_ns)
      : key_(key), event_generation_(event_generation), submit_ns_(submit_ns),
        deadline_ns_(deadline_ns) {}
  Status poison(CudaFrontierFailure failure, std::uint32_t device_error_code,
                const char* message);

  CudaErrorRecordKey key_;
  std::uint64_t event_generation_;
  std::uint64_t submit_ns_;
  std::uint64_t deadline_ns_;
  CudaFrontierFailure first_failure_ = CudaFrontierFailure::kNone;
  std::uint32_t first_device_error_code_ = 0;
  bool completed_ = false;
  bool poisoned_ = false;
  bool draining_ = false;
};

}  // namespace pih
