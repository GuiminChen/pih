#include "pih/model/qwen3_bf16_step_transaction.h"

#include <cstring>

namespace pih {

Status QwenBf16PreparedStepCompute::submit(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    DriverStreamHandle stream) {
  return execution_.run(clear_driver, kernel_driver, linear_driver, stream);
}

Status QwenBf16PreparedStepCompute::submit_instrumented_and_record(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    QwenBf16TapSnapshotDriver& snapshot_driver,
    const QwenBf16TapBindingPlan& binding_plan,
    QwenBf16TapSnapshotFrontierRecorder& frontier_recorder,
    CompletionEventDriver& event_driver,
    DriverStreamHandle stream,
    std::uint64_t submit_ns,
    std::uint64_t deadline_ns) {
  return execution_.run_instrumented_and_record(
      clear_driver, kernel_driver, linear_driver, snapshot_driver,
      binding_plan, frontier_recorder, event_driver, stream, submit_ns,
      deadline_ns);
}

Result<QwenBf16StepTransaction> QwenBf16StepTransaction::Create(
    QwenBf16StepUpload upload, QwenBf16StepCompute& compute,
    QwenBf16StepReadback readback, CompletionEventSlot completion_slot,
    CudaCompletionFrontier completion_frontier,
    std::span<std::byte> pinned_result_backing,
    DriverStreamHandle stream, std::uint64_t event_generation,
    QwenBf16StepHealthProvider& health_provider) {
  if (stream == 0 || event_generation == 0 ||
      upload.state() != QwenBf16StepUploadState::kPrepared ||
      readback.state() != QwenBf16StepReadbackState::kPrepared ||
      completion_slot.state() != CompletionEventSlotState::kIdle ||
      upload.stream() != stream || readback.stream() != stream ||
      upload.event_generation() != event_generation ||
      readback.event_generation() != event_generation ||
      completion_frontier.event_generation() != event_generation ||
      upload.context_identity() != readback.context_identity() ||
      upload.context_identity() != completion_slot.context_identity() ||
      pinned_result_backing.size() != QwenBf16StepResultLayout::kTotalBytes) {
    return Status::InvalidArgument(
        "Qwen step transaction identity or initial state is invalid");
  }
  const Status initialized =
      QwenBf16StepResultLayout::initialize(pinned_result_backing);
  if (!initialized.ok()) return initialized;
  return QwenBf16StepTransaction(
      std::move(upload), compute, std::move(readback),
      std::move(completion_slot), std::move(completion_frontier),
      pinned_result_backing, stream, event_generation, health_provider);
}

Status QwenBf16StepTransaction::submit(
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    CompletionEventDriver& event_driver) {
  if (state_ != QwenBf16StepTransactionState::kPrepared) {
    return Status::FailedPrecondition(
        "Qwen step transaction is not submit-ready");
  }
  Status status = upload_.submit(copy_driver);
  if (status.ok()) {
    status = compute_->submit(clear_driver, kernel_driver, linear_driver,
                              stream_);
  }
  if (status.ok()) status = readback_.submit(copy_driver);
  if (status.ok()) {
    status = completion_slot_.record(event_driver, stream_, event_generation_);
  }
  if (!status.ok()) {
    state_ = QwenBf16StepTransactionState::kPoisoned;
    return status;
  }
  state_ = QwenBf16StepTransactionState::kSubmitted;
  return Status::Ok();
}

Result<CompletionPublicationEvidence> QwenBf16StepTransaction::collect() {
  auto health = health_provider_->collect();
  if (!health.ok()) return health.status();
  std::uint32_t error = UINT32_MAX;
  std::memcpy(&error,
              pinned_result_backing_.data() +
                  QwenBf16StepResultLayout::device_error().offset_bytes,
              sizeof(error));
  return CompletionPublicationEvidence{
      health->submit_thread_last_error_clean, error,
      health->engine_poisoned};
}

Result<std::int64_t> QwenBf16StepTransaction::poll(
    CompletionEventDriver& event_driver) {
  if (state_ != QwenBf16StepTransactionState::kSubmitted) {
    return Status::FailedPrecondition(
        "Qwen step transaction is not awaiting completion");
  }
  const Status status = completion_slot_.poll(
      event_driver, completion_frontier_, *this);
  if (!status.ok()) {
    if (status.code() != StatusCode::kUnavailable) {
      state_ = QwenBf16StepTransactionState::kPoisoned;
    }
    return status;
  }
  auto token = QwenBf16StepResultLayout::parse(
      pinned_result_backing_, completion_frontier_.publication_authorized());
  if (!token.ok()) {
    state_ = QwenBf16StepTransactionState::kPoisoned;
    return token.status();
  }
  state_ = QwenBf16StepTransactionState::kCompleted;
  return token;
}

Status QwenBf16StepTransaction::expire(std::uint64_t now_ns) {
  if (state_ != QwenBf16StepTransactionState::kSubmitted) {
    return Status::FailedPrecondition(
        "Qwen step transaction is not awaiting completion");
  }
  const Status status = completion_frontier_.expire(now_ns);
  if (status.code() != StatusCode::kUnavailable) {
    state_ = QwenBf16StepTransactionState::kPoisoned;
  }
  return status;
}

Status QwenBf16StepTransaction::release_completion() {
  if (state_ != QwenBf16StepTransactionState::kCompleted) {
    return Status::FailedPrecondition(
        "Qwen step completion cannot be released before publication");
  }
  return completion_slot_.release(event_generation_);
}

}  // namespace pih
