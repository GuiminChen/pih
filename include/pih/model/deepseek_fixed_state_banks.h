#pragma once

#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/deepseek_expert_compute_arena.h"
#include "pih/model/deepseek_expert_subwave_executor.h"

namespace pih {

class DeepSeekFixedStateBankOperations {
 public:
  virtual ~DeepSeekFixedStateBankOperations() = default;
  virtual Status copy_d2d_async(std::uintptr_t destination,
                                std::uintptr_t source, std::uint64_t bytes,
                                std::uintptr_t stream) = 0;
  virtual Status record_event(std::uintptr_t event,
                              std::uintptr_t stream) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) = 0;
};

enum class DeepSeekFixedStateTransactionState : std::uint8_t {
  kIdle,
  kPrepared,
  kAwaitingCompletion,
  kReadyToResolve,
  kPoisoned,
};

class DeepSeekFixedStateBanks final {
 public:
  static Result<DeepSeekFixedStateBanks> Create(
      DeepSeekExpertArenaSpan first, DeepSeekExpertArenaSpan second,
      std::uintptr_t completion_event,
      DeepSeekFixedStateBankOperations& operations);

  Status validate_prepare(std::uintptr_t stream) const;
  Status prepare(std::uintptr_t stream);
  Status seal(std::uintptr_t stream);
  Result<DeepSeekExpertAsyncStatus> poll();
  Status validate_commit() const;
  Status commit();
  Status cancel_prepared();
  Status abort();

  [[nodiscard]] std::uintptr_t committed_address() const noexcept;
  [[nodiscard]] std::uintptr_t tentative_address() const noexcept;
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }
  [[nodiscard]] std::uint64_t bank_bytes() const noexcept {
    return banks_[0].bytes;
  }
  [[nodiscard]] DeepSeekFixedStateTransactionState state() const noexcept {
    return state_;
  }

 private:
  DeepSeekExpertArenaSpan banks_[2];
  std::uint8_t committed_bank_ = 0;
  std::uintptr_t completion_event_ = 0;
  DeepSeekFixedStateBankOperations* operations_ = nullptr;
  std::uint64_t generation_ = 1;
  DeepSeekFixedStateTransactionState state_ =
      DeepSeekFixedStateTransactionState::kIdle;
};

}  // namespace pih
