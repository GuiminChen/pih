#pragma once

#include <cstdint>

#include "pih/backend/cuda/completion_frontier.h"
#include "pih/backend/cuda/typed_copy_plan.h"

namespace pih {

enum class CudaStagingState : std::uint8_t {
  kFree = 0,
  kCpuFilling,
  kGpuFilling,
  kReadyToSubmit,
  kInFlight,
  kCompleteVerified,
  kConsumed,
  kSuspect,
};

class CudaStagingLease final {
 public:
  static Result<CudaStagingLease> Create(std::uint64_t owner_id,
                                         std::uint64_t capacity_bytes);

  Status begin_cpu_fill(std::uint64_t generation, std::uint64_t bytes);
  Status begin_gpu_fill(std::uint64_t generation, std::uint64_t bytes,
                        std::uint64_t producer_plan_id);
  Status mark_ready(std::uint64_t generation);
  Status complete_gpu_fill(const CudaCompletionFrontier& frontier);
  Status submit_copy(CudaTypedCopyPlan& plan, TypedCopyDriver& driver);
  Status complete_copy(const CudaCompletionFrontier& frontier);
  Status consume(std::uint64_t generation);
  Status release(std::uint64_t generation);
  Status fail(std::uint64_t generation);

  [[nodiscard]] CudaStagingState state() const noexcept { return state_; }
  [[nodiscard]] bool suspect() const noexcept {
    return state_ == CudaStagingState::kSuspect;
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

 private:
  CudaStagingLease(std::uint64_t owner_id, std::uint64_t capacity_bytes)
      : owner_id_(owner_id), capacity_bytes_(capacity_bytes) {}
  Status begin_fill(CudaStagingState filling_state, std::uint64_t generation,
                    std::uint64_t bytes);
  [[nodiscard]] bool generation_matches(std::uint64_t generation) const noexcept;

  std::uint64_t owner_id_;
  std::uint64_t capacity_bytes_;
  std::uint64_t generation_ = 0;
  std::uint64_t last_generation_ = 0;
  std::uint64_t bytes_ = 0;
  std::uint64_t copy_plan_id_ = 0;
  std::uint64_t producer_plan_id_ = 0;
  CudaStagingState state_ = CudaStagingState::kFree;
};

}  // namespace pih
