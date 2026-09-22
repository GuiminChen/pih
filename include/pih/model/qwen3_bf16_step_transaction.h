#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_bf16_prepared_execution.h"
#include "pih/model/qwen3_bf16_step_readback.h"
#include "pih/model/qwen3_bf16_step_upload.h"

namespace pih {

enum class QwenBf16StepTransactionState : std::uint8_t {
  kPrepared,
  kSubmitted,
  kCompleted,
  kPoisoned,
};

class QwenBf16StepCompute {
 public:
  virtual ~QwenBf16StepCompute() = default;
  virtual Status submit(QwenBf16DeviceErrorClearDriver& clear_driver,
                        KernelLaunchDriver& kernel_driver,
                        QwenBf16LinearExecutionDriver& linear_driver,
                        DriverStreamHandle stream) = 0;
};

class QwenBf16InstrumentedStepCompute {
 public:
  virtual ~QwenBf16InstrumentedStepCompute() = default;
  virtual Status submit_instrumented_and_record(
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenBf16LinearExecutionDriver& linear_driver,
      QwenBf16TapSnapshotDriver& snapshot_driver,
      const QwenBf16TapBindingPlan& binding_plan,
      QwenBf16TapSnapshotFrontierRecorder& frontier_recorder,
      CompletionEventDriver& event_driver,
      DriverStreamHandle stream,
      std::uint64_t submit_ns,
      std::uint64_t deadline_ns) = 0;
};

class QwenBf16PreparedStepCompute final
    : public QwenBf16StepCompute,
      public QwenBf16InstrumentedStepCompute {
 public:
  explicit QwenBf16PreparedStepCompute(QwenBf16PreparedExecution execution)
      : execution_(std::move(execution)) {}

  Status submit(QwenBf16DeviceErrorClearDriver& clear_driver,
                KernelLaunchDriver& kernel_driver,
                QwenBf16LinearExecutionDriver& linear_driver,
                DriverStreamHandle stream) override;

  Status submit_instrumented_and_record(
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenBf16LinearExecutionDriver& linear_driver,
      QwenBf16TapSnapshotDriver& snapshot_driver,
      const QwenBf16TapBindingPlan& binding_plan,
      QwenBf16TapSnapshotFrontierRecorder& frontier_recorder,
      CompletionEventDriver& event_driver,
      DriverStreamHandle stream,
      std::uint64_t submit_ns,
      std::uint64_t deadline_ns) override;

 private:
  QwenBf16PreparedExecution execution_;
};

struct QwenBf16StepHealth final {
  bool submit_thread_last_error_clean;
  bool engine_poisoned;
};

class QwenBf16StepHealthProvider {
 public:
  virtual ~QwenBf16StepHealthProvider() = default;
  virtual Result<QwenBf16StepHealth> collect() = 0;
};

class QwenBf16StepTransaction final : private CompletionEvidenceProvider {
 public:
  static Result<QwenBf16StepTransaction> Create(
      QwenBf16StepUpload upload, QwenBf16StepCompute& compute,
      QwenBf16StepReadback readback, CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result_backing,
      DriverStreamHandle stream, std::uint64_t event_generation,
      QwenBf16StepHealthProvider& health_provider);

  QwenBf16StepTransaction(const QwenBf16StepTransaction&) = delete;
  QwenBf16StepTransaction& operator=(const QwenBf16StepTransaction&) = delete;
  QwenBf16StepTransaction(QwenBf16StepTransaction&&) noexcept = default;
  QwenBf16StepTransaction& operator=(QwenBf16StepTransaction&&) = delete;

  Status submit(TypedCopyDriver& copy_driver,
                QwenBf16DeviceErrorClearDriver& clear_driver,
                KernelLaunchDriver& kernel_driver,
                QwenBf16LinearExecutionDriver& linear_driver,
                CompletionEventDriver& event_driver);
  Result<std::int64_t> poll(CompletionEventDriver& event_driver);
  Status expire(std::uint64_t now_ns);
  Status release_completion();

  [[nodiscard]] QwenBf16StepTransactionState state() const noexcept {
    return state_;
  }

 private:
  QwenBf16StepTransaction(
      QwenBf16StepUpload upload, QwenBf16StepCompute& compute,
      QwenBf16StepReadback readback, CompletionEventSlot completion_slot,
      CudaCompletionFrontier completion_frontier,
      std::span<std::byte> pinned_result_backing,
      DriverStreamHandle stream, std::uint64_t event_generation,
      QwenBf16StepHealthProvider& health_provider)
      : upload_(std::move(upload)),
        compute_(&compute),
        readback_(std::move(readback)),
        completion_slot_(std::move(completion_slot)),
        completion_frontier_(std::move(completion_frontier)),
        pinned_result_backing_(pinned_result_backing),
        stream_(stream),
        event_generation_(event_generation),
        health_provider_(&health_provider) {}

  Result<CompletionPublicationEvidence> collect() override;

  QwenBf16StepUpload upload_;
  QwenBf16StepCompute* compute_;
  QwenBf16StepReadback readback_;
  CompletionEventSlot completion_slot_;
  CudaCompletionFrontier completion_frontier_;
  std::span<std::byte> pinned_result_backing_;
  DriverStreamHandle stream_;
  std::uint64_t event_generation_;
  QwenBf16StepHealthProvider* health_provider_;
  QwenBf16StepTransactionState state_ =
      QwenBf16StepTransactionState::kPrepared;
};

}  // namespace pih
