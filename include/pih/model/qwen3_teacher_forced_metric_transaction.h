#pragma once

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_bf16_step_transaction.h"
#include "pih/model/qwen3_teacher_forced_metric_plan.h"
#include "pih/model/qwen3_teacher_forced_metric_result_layout.h"
#include "pih/model/qwen3_teacher_forced_metric_transfer.h"

namespace pih {

enum class QwenTeacherForcedMetricTransactionState : std::uint8_t {
  kPrepared, kSubmitted, kCompleted, kPoisoned,
};

class QwenTeacherForcedMetricTransaction final
    : private CompletionEvidenceProvider {
 public:
  static Result<QwenTeacherForcedMetricTransaction> Create(
      QwenTeacherForcedMetricTransfer transfer,
      QwenTeacherForcedMetricPlan metric_plan, TensorView device_error,
      QwenTeacherForcedMetricResultLayout result_layout,
      CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result, DriverStreamHandle stream,
      std::uint64_t event_generation, std::int32_t owning_rank,
      QwenBf16StepHealthProvider& health_provider);

  Status submit(TypedCopyDriver& copy_driver,
                QwenBf16DeviceErrorClearDriver& clear_driver,
                KernelLaunchDriver& kernel_driver,
                CompletionEventDriver& event_driver);
  Result<QwenTeacherForcedMetricBatch> poll(
      CompletionEventDriver& event_driver);
  Status expire(std::uint64_t now_ns);
  Status release_completion();
  [[nodiscard]] QwenTeacherForcedMetricTransactionState state() const noexcept {
    return state_;
  }

 private:
  QwenTeacherForcedMetricTransaction(
      QwenTeacherForcedMetricTransfer transfer,
      QwenTeacherForcedMetricPlan metric_plan, TensorView device_error,
      QwenTeacherForcedMetricResultLayout result_layout,
      CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result, DriverStreamHandle stream,
      std::uint64_t event_generation, std::int32_t owning_rank,
      QwenBf16StepHealthProvider& health_provider)
      : transfer_(std::move(transfer)), metric_plan_(std::move(metric_plan)),
        device_error_(device_error), result_layout_(result_layout),
        completion_slot_(std::move(completion_slot)),
        completion_frontier_(std::move(completion_frontier)),
        pinned_result_(pinned_result), stream_(stream),
        event_generation_(event_generation), owning_rank_(owning_rank),
        health_provider_(&health_provider) {}
  Result<CompletionPublicationEvidence> collect() override;

  QwenTeacherForcedMetricTransfer transfer_;
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
  QwenTeacherForcedMetricTransactionState state_ =
      QwenTeacherForcedMetricTransactionState::kPrepared;
};

}  // namespace pih
