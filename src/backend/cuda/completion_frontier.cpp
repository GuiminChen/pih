#include "pih/backend/cuda/completion_frontier.h"

#include <limits>

namespace pih {
namespace {
bool valid_phase(CudaCompletionPhase phase) {
  switch (phase) {
    case CudaCompletionPhase::kPrefill:
    case CudaCompletionPhase::kDecode:
    case CudaCompletionPhase::kKvAppend:
    case CudaCompletionPhase::kAttention:
    case CudaCompletionPhase::kCopy:
      return true;
  }
  return false;
}
}  // namespace

Result<CudaCompletionFrontier> CudaCompletionFrontier::Create(
    CudaErrorRecordKey key, std::uint64_t event_generation,
    std::uint64_t submit_ns, std::uint64_t deadline_ns) {
  if (key.epoch == 0 || key.rank == std::numeric_limits<std::uint32_t>::max() ||
      key.plan_generation == 0 || !valid_phase(key.phase) ||
      key.completion_frontier == 0 || event_generation == 0 ||
      deadline_ns <= submit_ns) {
    return Status::InvalidArgument("CUDA completion frontier identity is invalid");
  }
  return CudaCompletionFrontier(key, event_generation, submit_ns, deadline_ns);
}

Status CudaCompletionFrontier::poison(CudaFrontierFailure failure,
                                      std::uint32_t device_error_code,
                                      const char* message) {
  if (!poisoned_) {
    first_failure_ = failure;
    first_device_error_code_ = device_error_code;
  }
  poisoned_ = true;
  draining_ = true;
  return Status::Internal(message);
}

Status CudaCompletionFrontier::observe(
    std::uint64_t observed_event_generation,
    CudaEventQueryResult query_result, bool submit_thread_last_error_clean,
    std::uint32_t device_error_code, bool engine_poisoned) {
  if (completed_) return Status::FailedPrecondition("CUDA frontier already completed");
  if (poisoned_) return Status::FailedPrecondition("CUDA frontier is poisoned");
  if (observed_event_generation != event_generation_) {
    return poison(CudaFrontierFailure::kEventGenerationMismatch, 0,
                  "CUDA completion event generation mismatched");
  }
  if (query_result == CudaEventQueryResult::kNotReady) {
    if (device_error_code != 0 || engine_poisoned ||
        !submit_thread_last_error_clean) {
      return poison(CudaFrontierFailure::kInvalidObservation,
                    device_error_code,
                    "CUDA NotReady observation carried publish evidence");
    }
    return Status::Unavailable("CUDA completion event is not ready");
  }
  if (query_result == CudaEventQueryResult::kError) {
    return poison(CudaFrontierFailure::kUnexpectedQueryError,
                  device_error_code,
                  "CUDA completion query reported asynchronous error");
  }
  if (query_result != CudaEventQueryResult::kSuccess) {
    return poison(CudaFrontierFailure::kInvalidObservation, device_error_code,
                  "CUDA completion query result is invalid");
  }
  if (!submit_thread_last_error_clean) {
    return poison(CudaFrontierFailure::kLastErrorDirty, device_error_code,
                  "CUDA submit thread last-error state is dirty");
  }
  if (device_error_code != 0) {
    return poison(CudaFrontierFailure::kDeviceInvariant, device_error_code,
                  "CUDA kernel reported a device invariant failure");
  }
  if (engine_poisoned) {
    return poison(CudaFrontierFailure::kEnginePoisoned, 0,
                  "CUDA engine was poisoned before publication");
  }
  completed_ = true;
  return Status::Ok();
}

Status CudaCompletionFrontier::expire(std::uint64_t now_ns) {
  if (completed_) return Status::FailedPrecondition("completed CUDA frontier cannot expire");
  if (poisoned_) return Status::FailedPrecondition("CUDA frontier is poisoned");
  if (now_ns < submit_ns_) {
    return poison(CudaFrontierFailure::kInvalidObservation, 0,
                  "CUDA completion clock regressed");
  }
  if (now_ns < deadline_ns_) {
    return Status::Unavailable("CUDA completion deadline has not expired");
  }
  return poison(CudaFrontierFailure::kDeadlineExpired, 0,
                "CUDA completion deadline expired");
}

}  // namespace pih
