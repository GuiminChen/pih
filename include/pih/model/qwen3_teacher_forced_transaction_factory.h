#pragma once

#include <cstdint>

#include "pih/model/qwen3_teacher_forced_batch_plan.h"
#include "pih/model/qwen3_teacher_forced_chunk_transaction.h"
#include "pih/model/qwen3_teacher_forced_metric_arenas.h"
#include "pih/model/qwen3_teacher_forced_transaction_identity.h"

namespace pih {

struct QwenTeacherForcedMetricOwnerIds final {
  std::uint64_t pinned_sample_rows = 0;
  std::uint64_t device_sample_rows = 0;
  std::uint64_t pinned_targets = 0;
  std::uint64_t device_targets = 0;
  std::uint64_t device_result = 0;
  std::uint64_t pinned_result = 0;
};

struct QwenTeacherForcedTransactionFactoryCapability final {
  std::uint64_t arena_identity = 0;
  std::uint64_t factory_generation = 0;
  friend bool operator==(
      const QwenTeacherForcedTransactionFactoryCapability&,
      const QwenTeacherForcedTransactionFactoryCapability&) = default;
};

class QwenTeacherForcedTransactionFactory final {
 public:
  static Result<QwenTeacherForcedTransactionFactory> Create(
      QwenTeacherForcedMetricArenas& arenas,
      ResolvedKernelFunction gather_function,
      ResolvedKernelFunction metric_function,
      TensorView normalized_rows, TensorView gathered_rows,
      TensorView lm_head_weight, TensorView logits,
      std::uintptr_t context_identity, DriverStreamHandle stream,
      std::int32_t owning_rank, QwenTeacherForcedMetricOwnerIds owner_ids);

  Result<QwenTeacherForcedChunkTransaction> build(
      const QwenTeacherForcedBatchPlan& batch,
      std::uint64_t event_generation,
      std::uint64_t row_upload_plan_id,
      std::uint64_t target_upload_plan_id,
      std::uint64_t readback_plan_id,
      CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      QwenBf16StepHealthProvider& health_provider);
  Result<QwenTeacherForcedChunkTransaction> build_reserved(
      const QwenTeacherForcedBatchPlan& batch,
      const QwenTeacherForcedTransactionIdentityReservation& identity,
      QwenTeacherForcedTransactionCompletion completion,
      QwenBf16StepHealthProvider& health_provider);
  [[nodiscard]] QwenTeacherForcedTransactionFactoryCapability capability()
      const noexcept {
    return capability_;
  }

 private:
  QwenTeacherForcedTransactionFactory(
      QwenTeacherForcedMetricArenas& arenas,
      ResolvedKernelFunction gather_function,
      ResolvedKernelFunction metric_function,
      TensorView normalized_rows, TensorView gathered_rows,
      TensorView lm_head_weight, TensorView logits,
      std::uintptr_t context_identity, DriverStreamHandle stream,
      std::int32_t owning_rank, QwenTeacherForcedMetricOwnerIds owner_ids,
      QwenTeacherForcedTransactionFactoryCapability capability)
      : arenas_(&arenas), gather_function_(std::move(gather_function)),
        metric_function_(std::move(metric_function)),
        normalized_rows_(normalized_rows), gathered_rows_(gathered_rows),
        lm_head_weight_(lm_head_weight), logits_(logits),
        context_identity_(context_identity), stream_(stream),
        owning_rank_(owning_rank), owner_ids_(owner_ids),
        capability_(capability) {}

  QwenTeacherForcedMetricArenas* arenas_ = nullptr;
  ResolvedKernelFunction gather_function_;
  ResolvedKernelFunction metric_function_;
  TensorView normalized_rows_;
  TensorView gathered_rows_;
  TensorView lm_head_weight_;
  TensorView logits_;
  std::uintptr_t context_identity_ = 0;
  DriverStreamHandle stream_ = 0;
  std::int32_t owning_rank_ = -1;
  QwenTeacherForcedMetricOwnerIds owner_ids_{};
  QwenTeacherForcedTransactionFactoryCapability capability_{};
};

}  // namespace pih
