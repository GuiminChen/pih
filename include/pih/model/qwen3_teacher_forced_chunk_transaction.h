#pragma once

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_bf16_step_transaction.h"
#include "pih/model/qwen3_teacher_forced_logits_plan.h"
#include "pih/model/qwen3_teacher_forced_metric_plan.h"
#include "pih/model/qwen3_teacher_forced_metric_arena_lease.h"
#include "pih/model/qwen3_teacher_forced_metric_result_layout.h"
#include "pih/model/qwen3_teacher_forced_metric_transfer.h"

namespace pih {

enum class QwenTeacherForcedChunkTransactionState : std::uint8_t {
  kPrepared, kSubmitted, kCompleted, kPoisoned,
};

enum class QwenTeacherForcedChunkRole : std::uint8_t {
  kNone, kBf16, kInt4,
};

class QwenTeacherForcedChunkTransaction final
    : private CompletionEvidenceProvider {
 public:
  static Result<QwenTeacherForcedChunkTransaction> Create(
      QwenTeacherForcedMetricTransfer transfer,
      QwenTeacherForcedLogitsPlan logits_plan,
      QwenTeacherForcedMetricPlan metric_plan, TensorView device_error,
      QwenTeacherForcedMetricResultLayout result_layout,
      CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result, DriverStreamHandle stream,
      std::uint64_t event_generation, std::int32_t owning_rank,
      QwenBf16StepHealthProvider& health_provider,
      QwenTeacherForcedMetricArenaLease arena_lease = {});

  Status submit_bf16(TypedCopyDriver& copy_driver,
                     QwenBf16DeviceErrorClearDriver& clear_driver,
                     KernelLaunchDriver& kernel_driver,
                     QwenBf16LinearExecutionDriver& head_driver,
                     CompletionEventDriver& event_driver);
  Status submit_int4(TypedCopyDriver& copy_driver,
                     QwenBf16DeviceErrorClearDriver& clear_driver,
                     KernelLaunchDriver& kernel_driver,
                     QwenInt4LmHeadExecutionDriver& head_driver,
                     CompletionEventDriver& event_driver);
  Result<QwenTeacherForcedMetricBatch> poll(
      CompletionEventDriver& event_driver);
  Status expire(std::uint64_t now_ns);
  Status release_completion();
  [[nodiscard]] QwenTeacherForcedChunkTransactionState state() const noexcept {
    return state_;
  }
  [[nodiscard]] QwenTeacherForcedChunkRole submitted_role() const noexcept {
    return submitted_role_;
  }

 private:
  QwenTeacherForcedChunkTransaction(
      QwenTeacherForcedMetricTransfer transfer,
      QwenTeacherForcedLogitsPlan logits_plan,
      QwenTeacherForcedMetricPlan metric_plan, TensorView device_error,
      QwenTeacherForcedMetricResultLayout result_layout,
      CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result, DriverStreamHandle stream,
      std::uint64_t event_generation, std::int32_t owning_rank,
      QwenBf16StepHealthProvider& health_provider,
      QwenTeacherForcedMetricArenaLease arena_lease)
      : transfer_(std::move(transfer)), logits_plan_(std::move(logits_plan)),
        metric_plan_(std::move(metric_plan)), device_error_(device_error),
        result_layout_(result_layout),
        completion_slot_(std::move(completion_slot)),
        completion_frontier_(std::move(completion_frontier)),
        pinned_result_(pinned_result), stream_(stream),
        event_generation_(event_generation), owning_rank_(owning_rank),
        health_provider_(&health_provider), arena_lease_(std::move(arena_lease)) {}

  Result<CompletionPublicationEvidence> collect() override;

  QwenTeacherForcedMetricTransfer transfer_;
  QwenTeacherForcedLogitsPlan logits_plan_;
  QwenTeacherForcedMetricPlan metric_plan_;
  TensorView device_error_;
  QwenTeacherForcedMetricResultLayout result_layout_;
  CompletionEventSlot completion_slot_;
  CudaCompletionFrontier completion_frontier_;
  std::span<std::byte> pinned_result_;
  DriverStreamHandle stream_ = 0;
  std::uint64_t event_generation_ = 0;
  std::int32_t owning_rank_ = -1;
  QwenBf16StepHealthProvider* health_provider_ = nullptr;
  QwenTeacherForcedMetricArenaLease arena_lease_;
  QwenTeacherForcedChunkTransactionState state_ =
      QwenTeacherForcedChunkTransactionState::kPrepared;
  QwenTeacherForcedChunkRole submitted_role_ =
      QwenTeacherForcedChunkRole::kNone;
};

}  // namespace pih
