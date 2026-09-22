#pragma once

#include <array>
#include <cstdint>

#include "pih/core/result.h"

namespace pih {

enum class DeepSeekBoundaryCreditState : std::uint8_t {
  kFree,
  kPrepared,
  kCommitted,
  kCompleteVerified,
  kSuspect,
};

struct DeepSeekBoundaryCreditHandle final {
  std::uint32_t credit_index = 0;
  std::uint64_t credit_generation = 0;
  std::uint64_t pipeline_plan_sequence = 0;
  std::uint64_t operation_ordinal = 0;
};

class DeepSeekBoundaryCreditTracker final {
 public:
  static constexpr std::uint32_t kCreditCount = 2;
  static Result<DeepSeekBoundaryCreditTracker> Create(
      std::uint32_t credit_count);

  Result<DeepSeekBoundaryCreditHandle> reserve(
      std::uint64_t pipeline_plan_sequence,
      std::uint64_t operation_ordinal);
  Status abort_prepare(const DeepSeekBoundaryCreditHandle& handle);
  Status commit(const DeepSeekBoundaryCreditHandle& handle);
  Status complete_verified(const DeepSeekBoundaryCreditHandle& handle);
  Status release(const DeepSeekBoundaryCreditHandle& handle);
  Status mark_suspect(const DeepSeekBoundaryCreditHandle& handle);
  Status validate(const DeepSeekBoundaryCreditHandle& handle,
                  DeepSeekBoundaryCreditState expected) const {
    return require(handle, expected);
  }

  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] DeepSeekBoundaryCreditState state(
      std::uint32_t credit_index) const noexcept {
    return credit_index < kCreditCount
               ? slots_[credit_index].state
               : DeepSeekBoundaryCreditState::kSuspect;
  }

 private:
  struct Slot final {
    DeepSeekBoundaryCreditState state = DeepSeekBoundaryCreditState::kFree;
    std::uint64_t generation = 0;
    std::uint64_t plan_sequence = 0;
    std::uint64_t operation_ordinal = 0;
  };

  Status require(const DeepSeekBoundaryCreditHandle& handle,
                 DeepSeekBoundaryCreditState expected) const;
  void clear(Slot& slot) noexcept;

  std::array<Slot, kCreditCount> slots_{};
  bool poisoned_ = false;
};

}  // namespace pih
