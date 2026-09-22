#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include "pih/model/qwen3_teacher_forced_pair_run_executor.h"
#include "pih/model/qwen3_teacher_forced_transaction_factory.h"

namespace pih {

struct QwenTeacherForcedPairRunIdentity final {
  std::uint64_t engine_epoch = 0;
  std::uint32_t rank = 0;
  DriverEventHandle event = 0;
  std::uintptr_t context_identity = 0;
  std::uint64_t first_plan_id = 0;
  std::uint64_t first_event_generation = 0;
  std::uint64_t first_completion_frontier = 0;
};

enum class QwenTeacherForcedPairRunDriverPhase : std::uint8_t {
  kAwaitingBf16 = 0,
  kBf16Pending,
  kAwaitingInt4,
  kInt4Pending,
  kFinalized,
  kPoisoned,
};

class QwenTeacherForcedPairRunDriver final {
 public:
  static Result<QwenTeacherForcedPairRunDriver> Create(
      std::span<const QwenTeacherForcedChunk> expected_chunks,
      QwenTeacherForcedPairRunIdentity identity);

  Status submit_bf16(
      const QwenTeacherForcedBatchPlan& batch,
      QwenTeacherForcedTransactionFactory& factory,
      std::uint64_t submit_ns, std::uint64_t deadline_ns,
      QwenBf16StepHealthProvider& health_provider,
      TypedCopyDriver& copy_driver,
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenBf16LinearExecutionDriver& head_driver,
      CompletionEventDriver& event_driver);
  Status submit_int4(
      const QwenTeacherForcedBatchPlan& batch,
      QwenTeacherForcedTransactionFactory& factory,
      std::uint64_t submit_ns, std::uint64_t deadline_ns,
      QwenBf16StepHealthProvider& health_provider,
      TypedCopyDriver& copy_driver,
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenInt4LmHeadExecutionDriver& head_driver,
      CompletionEventDriver& event_driver);
  Status poll(CompletionEventDriver& event_driver);
  Status expire(std::uint64_t now_ns);
  Result<std::array<QwenTeacherForcedCategoryAggregate, 6>> finalize();

  [[nodiscard]] std::uint64_t next_plan_id() const noexcept {
    return next_plan_id_;
  }
  [[nodiscard]] std::uint64_t next_event_generation() const noexcept {
    return next_event_generation_;
  }
  [[nodiscard]] std::uint64_t next_completion_frontier() const noexcept {
    return next_completion_frontier_;
  }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] QwenTeacherForcedPairRunDriverPhase phase() const noexcept {
    return phase_;
  }

 private:
  QwenTeacherForcedPairRunDriver(
      QwenTeacherForcedPairRunExecutor executor,
      QwenTeacherForcedPairRunIdentity identity)
      : executor_(std::move(executor)), identity_(identity),
        next_plan_id_(identity.first_plan_id),
        next_event_generation_(identity.first_event_generation),
        next_completion_frontier_(identity.first_completion_frontier) {}
  Result<QwenTeacherForcedChunkTransaction> build(
      const QwenTeacherForcedBatchPlan& batch,
      QwenTeacherForcedTransactionFactory& factory,
      std::uint64_t submit_ns, std::uint64_t deadline_ns,
      QwenBf16StepHealthProvider& health_provider,
      QwenTeacherForcedTransactionIdentityReservation* reservation,
      std::uint64_t* next_frontier);
  Status validate_batch(const QwenTeacherForcedBatchPlan& batch) const;
  Status validate_factory(
      const QwenTeacherForcedTransactionFactory& factory) const;
  void commit(const QwenTeacherForcedTransactionIdentityReservation& value,
              std::uint64_t next_frontier) noexcept;

  QwenTeacherForcedPairRunExecutor executor_;
  QwenTeacherForcedPairRunIdentity identity_{};
  std::uint64_t next_plan_id_ = 0;
  std::uint64_t next_event_generation_ = 0;
  std::uint64_t next_completion_frontier_ = 0;
  std::optional<QwenTeacherForcedTransactionFactoryCapability>
      factory_capability_;
  QwenTeacherForcedPairRunDriverPhase phase_ =
      QwenTeacherForcedPairRunDriverPhase::kAwaitingBf16;
  bool poisoned_ = false;
};

}  // namespace pih
