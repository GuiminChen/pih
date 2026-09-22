#pragma once

#include <cstdint>
#include <vector>

#include "pih/model/deepseek_pipeline_transaction.h"

namespace pih {

enum class DeepSeekPipelineCoordinatorState : std::uint8_t {
  kPreparing,
  kReadyToCommit,
  kCommitted,
  kDraining,
  kComplete,
  kAborted,
  kPoisoned,
};

class DeepSeekPipelineCoordinator final {
 public:
  static Result<DeepSeekPipelineCoordinator> Create(
      DeepSeekPipelineTransaction& transaction, std::uint32_t world_size);

  Status stage_ready(std::uint32_t rank);
  [[nodiscard]] Status validate_stage_reject(std::uint32_t rank) const;
  Status stage_reject(std::uint32_t rank, Status reason);
  [[nodiscard]] Status validate_commit() const;
  Status commit();
  Status cancel_after_commit();
  [[nodiscard]] Status validate_stage_complete(std::uint32_t rank) const;
  Status stage_complete(std::uint32_t rank);
  [[nodiscard]] Status validate_stage_failed(std::uint32_t rank) const;
  Status stage_failed(std::uint32_t rank, Status reason);

  [[nodiscard]] DeepSeekPipelineCoordinatorState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint32_t ready_count() const noexcept {
    return ready_count_;
  }
  [[nodiscard]] std::uint32_t complete_count() const noexcept {
    return complete_count_;
  }
  [[nodiscard]] bool suppress_output() const noexcept {
    return suppress_output_;
  }

 private:
  DeepSeekPipelineCoordinator(DeepSeekPipelineTransaction& transaction,
                              std::uint32_t world_size)
      : transaction_(&transaction), world_size_(world_size),
        ready_(world_size, false), complete_(world_size, false) {}
  Status validate_rank(std::uint32_t rank) const;
  Status poison(Status reason);

  DeepSeekPipelineTransaction* transaction_ = nullptr;
  std::uint32_t world_size_ = 0;
  std::vector<bool> ready_;
  std::vector<bool> complete_;
  std::uint32_t ready_count_ = 0;
  std::uint32_t complete_count_ = 0;
  bool suppress_output_ = false;
  DeepSeekPipelineCoordinatorState state_ =
      DeepSeekPipelineCoordinatorState::kPreparing;
};

}  // namespace pih
