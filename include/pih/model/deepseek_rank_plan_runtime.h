#pragma once

#include <memory>

#include "pih/model/deepseek_pipeline_stage_executor.h"
#include "pih/model/deepseek_rank_plan_reservation.h"

namespace pih {


struct DeepSeekOwnedStageExecutionDrivers final {
  std::unique_ptr<DeepSeekStageComputeDriver> compute;
};


enum class DeepSeekRankPlanRuntimeState : std::uint8_t {
  kPrepared,
  kReady,
  kCommitted,
  kExecuting,
  kCompletionReady,
  kComplete,
  kPoisoned,
};

class DeepSeekRankPlanRuntime final {
 public:
  static Result<DeepSeekRankPlanRuntime> CreateBorrowedCompute(
      DeepSeekPipelineTransaction transaction,
      DeepSeekStagePlan stage,
      DeepSeekRankPlanReservation reservation,
      DeepSeekStageComputeDriver& compute);

  DeepSeekRankPlanRuntime(const DeepSeekRankPlanRuntime&) = delete;
  DeepSeekRankPlanRuntime& operator=(const DeepSeekRankPlanRuntime&) = delete;
  DeepSeekRankPlanRuntime(DeepSeekRankPlanRuntime&&) noexcept = default;
  DeepSeekRankPlanRuntime& operator=(DeepSeekRankPlanRuntime&& other) noexcept;

  Status mark_ready();
  Status prepare_commit();
  [[nodiscard]] Status validate_commit() const;
  Status commit();
  Status cancel();
  Status advance_staged();
  [[nodiscard]] Status validate_complete() const;
  Status complete();
  Status advance();
  [[nodiscard]] bool cancellation_requested() const noexcept {
    return cancellation_requested_;
  }

  [[nodiscard]] DeepSeekRankPlanRuntimeState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint32_t rank() const noexcept {
    return stage_.rank;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] const DeepSeekPipelinePlanDescriptor& descriptor()
      const noexcept { return transaction_->descriptor(); }

 private:
  DeepSeekRankPlanRuntime(
      std::unique_ptr<DeepSeekPipelineTransaction> transaction,
      DeepSeekRankPlanReservation reservation,
      DeepSeekStageComputeDriver& compute, DeepSeekStagePlan stage,
      std::uint32_t world_size) noexcept
      : transaction_(std::move(transaction)),
        reservation_(std::move(reservation)),
        compute_(&compute), stage_(stage), world_size_(world_size) {}

  DeepSeekStageExecutionDrivers borrowed_drivers() noexcept;
  Status poison(Status cause) noexcept;

  // The transaction must outlive executor_, which stores a non-owning pointer.
  std::unique_ptr<DeepSeekPipelineTransaction> transaction_;
  DeepSeekRankPlanReservation reservation_;
  DeepSeekStageComputeDriver* compute_ = nullptr;
  DeepSeekStagePlan stage_;
  std::uint32_t world_size_ = 0;
  std::unique_ptr<DeepSeekPipelineStageExecutor> executor_;
  DeepSeekRankPlanRuntimeState state_ =
      DeepSeekRankPlanRuntimeState::kPrepared;
  bool cancellation_requested_ = false;
};

}  // namespace pih
