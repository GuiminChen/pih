#pragma once

#include <span>
#include <vector>

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_bf16_packed_readback.h"
#include "pih/model/qwen3_bf16_packed_step_upload.h"
#include "pih/model/qwen3_bf16_step_transaction.h"
#include "pih/model/qwen3_int4_packed_prepared_execution.h"

namespace pih {

enum class QwenInt4PackedStepTransactionState : std::uint8_t {
  kPrepared=1,kSubmitted=2,kCompleted=3,kPoisoned=4,
};

class QwenInt4PackedStepTransaction final
    : private CompletionEvidenceProvider {
 public:
  static Result<QwenInt4PackedStepTransaction> Create(
      QwenBf16PackedStepUpload upload,
      QwenInt4PackedPreparedExecution& compute,
      QwenBf16PackedReadback readback,
      QwenBf16PackedResultLayout result_layout,std::uint32_t sample_count,
      CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result_backing,
      DriverStreamHandle stream,std::uint64_t event_generation,
      QwenBf16StepHealthProvider& health_provider);

  QwenInt4PackedStepTransaction(const QwenInt4PackedStepTransaction&)=delete;
  QwenInt4PackedStepTransaction& operator=(
      const QwenInt4PackedStepTransaction&)=delete;
  QwenInt4PackedStepTransaction(QwenInt4PackedStepTransaction&&) noexcept=default;

  Status submit(TypedCopyDriver& copy_driver,
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenInt4LmHeadExecutionDriver& lm_head_driver,
      CompletionEventDriver& event_driver);
  Result<std::vector<QwenBf16PackedSampleReceipt>> poll_sampling(
      CompletionEventDriver& event_driver);
  Status expire(std::uint64_t now_ns);
  Status release_completion();
  [[nodiscard]] QwenInt4PackedStepTransactionState state() const noexcept{
    return state_;}

 private:
  QwenInt4PackedStepTransaction(QwenBf16PackedStepUpload upload,
      QwenInt4PackedPreparedExecution& compute,
      QwenBf16PackedReadback readback,QwenBf16PackedResultLayout layout,
      std::uint32_t sample_count,CompletionEventSlot slot,
      CudaCompletionFrontier frontier,std::span<std::byte> result,
      DriverStreamHandle stream,std::uint64_t event_generation,
      QwenBf16StepHealthProvider& health)
      :upload_(std::move(upload)),compute_(&compute),readback_(std::move(readback)),
       result_layout_(layout),sample_count_(sample_count),
       completion_slot_(std::move(slot)),completion_frontier_(std::move(frontier)),
       pinned_result_backing_(result),stream_(stream),
       event_generation_(event_generation),health_provider_(&health){}
  Result<CompletionPublicationEvidence> collect() override;

  QwenBf16PackedStepUpload upload_;
  QwenInt4PackedPreparedExecution* compute_;
  QwenBf16PackedReadback readback_;
  QwenBf16PackedResultLayout result_layout_;
  std::uint32_t sample_count_=0;
  CompletionEventSlot completion_slot_;
  CudaCompletionFrontier completion_frontier_;
  std::span<std::byte> pinned_result_backing_;
  DriverStreamHandle stream_=0;
  std::uint64_t event_generation_=0;
  QwenBf16StepHealthProvider* health_provider_;
  QwenInt4PackedStepTransactionState state_=
      QwenInt4PackedStepTransactionState::kPrepared;
};

}  // namespace pih
