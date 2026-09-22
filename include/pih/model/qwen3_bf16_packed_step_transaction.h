#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_bf16_packed_readback.h"
#include "pih/model/qwen3_bf16_packed_step_upload.h"
#include "pih/model/qwen3_bf16_step_transaction.h"

namespace pih {

enum class QwenBf16PackedStepTransactionState : std::uint8_t {
  kPrepared = 1,
  kSubmitted = 2,
  kCompleted = 3,
  kPoisoned = 4,
};

class QwenBf16PackedStepTransaction final
    : private CompletionEvidenceProvider {
 public:
  static Result<QwenBf16PackedStepTransaction> Create(
      QwenBf16PackedStepUpload upload, QwenBf16StepCompute& compute,
      QwenBf16PackedReadback readback,
      QwenBf16PackedResultLayout result_layout, std::uint32_t sample_count,
      CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result_backing,
      DriverStreamHandle stream, std::uint64_t event_generation,
      QwenBf16StepHealthProvider& health_provider);

  QwenBf16PackedStepTransaction(const QwenBf16PackedStepTransaction&) = delete;
  QwenBf16PackedStepTransaction& operator=(
      const QwenBf16PackedStepTransaction&) = delete;
  QwenBf16PackedStepTransaction(QwenBf16PackedStepTransaction&&) noexcept =
      default;

  Status submit(TypedCopyDriver& copy_driver,
                QwenBf16DeviceErrorClearDriver& clear_driver,
                KernelLaunchDriver& kernel_driver,
                QwenBf16LinearExecutionDriver& linear_driver,
                CompletionEventDriver& event_driver);
  Result<std::vector<std::uint32_t>> poll(
      CompletionEventDriver& event_driver);
  Result<std::vector<QwenBf16PackedSampleReceipt>> poll_sampling(
      CompletionEventDriver& event_driver);
  Status expire(std::uint64_t now_ns);
  Status release_completion();

  [[nodiscard]] QwenBf16PackedStepTransactionState state() const noexcept {
    return state_;
  }

 private:
  QwenBf16PackedStepTransaction(
      QwenBf16PackedStepUpload upload, QwenBf16StepCompute& compute,
      QwenBf16PackedReadback readback,
      QwenBf16PackedResultLayout result_layout, std::uint32_t sample_count,
      CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result_backing,
      DriverStreamHandle stream, std::uint64_t event_generation,
      QwenBf16StepHealthProvider& health_provider)
      : upload_(std::move(upload)), compute_(&compute),
        readback_(std::move(readback)), result_layout_(result_layout),
        sample_count_(sample_count), completion_slot_(std::move(completion_slot)),
        completion_frontier_(std::move(completion_frontier)),
        pinned_result_backing_(pinned_result_backing), stream_(stream),
        event_generation_(event_generation), health_provider_(&health_provider) {}

  Result<CompletionPublicationEvidence> collect() override;

  QwenBf16PackedStepUpload upload_;
  QwenBf16StepCompute* compute_;
  QwenBf16PackedReadback readback_;
  QwenBf16PackedResultLayout result_layout_;
  std::uint32_t sample_count_ = 0;
  CompletionEventSlot completion_slot_;
  CudaCompletionFrontier completion_frontier_;
  std::span<std::byte> pinned_result_backing_;
  DriverStreamHandle stream_ = 0;
  std::uint64_t event_generation_ = 0;
  QwenBf16StepHealthProvider* health_provider_;
  QwenBf16PackedStepTransactionState state_ =
      QwenBf16PackedStepTransactionState::kPrepared;
};

}  // namespace pih
