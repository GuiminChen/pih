#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_bf16_step_readback.h"
#include "pih/model/qwen3_bf16_step_transaction.h"
#include "pih/model/qwen3_bf16_step_upload.h"
#include "pih/model/qwen3_int4_prepared_execution.h"

namespace pih {

enum class QwenInt4StepTransactionState : std::uint8_t {
  kPrepared, kSubmitted, kCompleted, kPoisoned,
};

class QwenInt4StepCompute {
 public:
  virtual ~QwenInt4StepCompute() = default;
  virtual Status submit(QwenBf16DeviceErrorClearDriver& clear_driver,
                        KernelLaunchDriver& kernel_driver,
                        QwenInt4LmHeadExecutionDriver& lm_head_driver,
                        DriverStreamHandle stream) = 0;
};

class QwenInt4PreparedStepCompute final : public QwenInt4StepCompute {
 public:
  explicit QwenInt4PreparedStepCompute(QwenInt4PreparedExecution execution)
      : execution_(std::move(execution)) {}
  Status submit(QwenBf16DeviceErrorClearDriver& clear_driver,
                KernelLaunchDriver& kernel_driver,
                QwenInt4LmHeadExecutionDriver& lm_head_driver,
                DriverStreamHandle stream) override;
 private:
  QwenInt4PreparedExecution execution_;
};

class QwenInt4StepTransaction final : private CompletionEvidenceProvider {
 public:
  static Result<QwenInt4StepTransaction> Create(
      QwenBf16StepUpload upload, QwenInt4StepCompute& compute,
      QwenBf16StepReadback readback, CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result_backing,
      DriverStreamHandle stream, std::uint64_t event_generation,
      QwenBf16StepHealthProvider& health_provider);

  QwenInt4StepTransaction(const QwenInt4StepTransaction&) = delete;
  QwenInt4StepTransaction& operator=(const QwenInt4StepTransaction&) = delete;
  QwenInt4StepTransaction(QwenInt4StepTransaction&&) noexcept = default;
  QwenInt4StepTransaction& operator=(QwenInt4StepTransaction&&) = delete;

  Status submit(TypedCopyDriver& copy_driver,
                QwenBf16DeviceErrorClearDriver& clear_driver,
                KernelLaunchDriver& kernel_driver,
                QwenInt4LmHeadExecutionDriver& lm_head_driver,
                CompletionEventDriver& event_driver);
  Result<std::int64_t> poll(CompletionEventDriver& event_driver);
  Status expire(std::uint64_t now_ns);
  Status release_completion();
  [[nodiscard]] QwenInt4StepTransactionState state() const noexcept {
    return state_;
  }

 private:
  QwenInt4StepTransaction(
      QwenBf16StepUpload upload, QwenInt4StepCompute& compute,
      QwenBf16StepReadback readback, CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result_backing,
      DriverStreamHandle stream, std::uint64_t event_generation,
      QwenBf16StepHealthProvider& health_provider)
      : upload_(std::move(upload)), compute_(&compute),
        readback_(std::move(readback)), completion_slot_(std::move(completion_slot)),
        completion_frontier_(std::move(completion_frontier)),
        pinned_result_backing_(pinned_result_backing), stream_(stream),
        event_generation_(event_generation), health_provider_(&health_provider) {}
  Result<CompletionPublicationEvidence> collect() override;
  QwenBf16StepUpload upload_;
  QwenInt4StepCompute* compute_;
  QwenBf16StepReadback readback_;
  CompletionEventSlot completion_slot_;
  CudaCompletionFrontier completion_frontier_;
  std::span<std::byte> pinned_result_backing_;
  DriverStreamHandle stream_;
  std::uint64_t event_generation_;
  QwenBf16StepHealthProvider* health_provider_;
  QwenInt4StepTransactionState state_=QwenInt4StepTransactionState::kPrepared;
};

}  // namespace pih
