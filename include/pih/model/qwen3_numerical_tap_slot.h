#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/backend/cuda/completion_frontier.h"
#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/core/sha256.h"

namespace pih {

enum class QwenNumericalTapSlotState : std::uint8_t {
  kFree = 0,
  kProducerPending,
  kCopyReady,
  kCopyInFlight,
  kCopyComplete,
  kSealed,
  kPoisoned,
};

struct QwenNumericalTapReceipt final {
  std::size_t capture_index;
  std::uint64_t capture_generation;
  std::uint64_t bytes;
  Sha256Digest digest;
};

class QwenNumericalTapSlot final {
 public:
  static Result<QwenNumericalTapSlot> Create(std::uint64_t owner_id,
                                              std::uint64_t allocation_generation,
                                              std::uint64_t capacity_bytes);

  Status begin(std::uint64_t capture_generation, std::size_t capture_index,
               std::uint64_t source_owner_id,
               std::uint64_t source_generation, std::uint64_t bytes,
               std::uint64_t producer_plan_generation);
  Status complete_producer(const CudaCompletionFrontier& frontier);
  Status submit_copy(CudaTypedCopyPlan& copy, TypedCopyDriver& driver);
  Status complete_copy(const CudaCompletionFrontier& frontier);
  Result<QwenNumericalTapReceipt> seal(
      std::uint64_t capture_generation,
      std::span<const std::byte> observed_bytes);
  Status release(std::uint64_t capture_generation);

  [[nodiscard]] QwenNumericalTapSlotState state() const noexcept {
    return state_;
  }

 private:
  QwenNumericalTapSlot(std::uint64_t owner_id,
                       std::uint64_t allocation_generation,
                       std::uint64_t capacity_bytes)
      : owner_id_(owner_id), allocation_generation_(allocation_generation),
        capacity_bytes_(capacity_bytes) {}
  Status poison(const char* message);

  std::uint64_t owner_id_;
  std::uint64_t allocation_generation_;
  std::uint64_t capacity_bytes_;
  std::uint64_t last_generation_ = 0;
  std::uint64_t generation_ = 0;
  std::size_t capture_index_ = 0;
  std::uint64_t source_owner_id_ = 0;
  std::uint64_t source_generation_ = 0;
  std::uint64_t bytes_ = 0;
  std::uint64_t producer_plan_generation_ = 0;
  std::uint64_t copy_plan_id_ = 0;
  std::uint64_t copy_event_generation_ = 0;
  QwenNumericalTapSlotState state_ = QwenNumericalTapSlotState::kFree;
};

}  // namespace pih
