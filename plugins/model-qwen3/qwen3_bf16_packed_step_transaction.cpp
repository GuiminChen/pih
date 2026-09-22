#include "pih/model/qwen3_bf16_packed_step_transaction.h"

#include <cstring>

namespace pih {

Result<QwenBf16PackedStepTransaction> QwenBf16PackedStepTransaction::Create(
    QwenBf16PackedStepUpload upload, QwenBf16StepCompute& compute,
    QwenBf16PackedReadback readback,
    QwenBf16PackedResultLayout result_layout, std::uint32_t sample_count,
    CompletionEventSlot completion_slot,
    CudaCompletionFrontier completion_frontier,
    std::span<std::byte> pinned_result_backing,
    DriverStreamHandle stream, std::uint64_t event_generation,
    QwenBf16StepHealthProvider& health_provider) {
  if (sample_count > result_layout.sample_capacity() ||
      stream == 0 || event_generation == 0 ||
      upload.state() != QwenBf16PackedStepUploadState::kPrepared ||
      readback.state() != QwenBf16PackedReadbackState::kPrepared ||
      completion_slot.state() != CompletionEventSlotState::kIdle ||
      upload.stream() != stream || readback.stream() != stream ||
      upload.event_generation() != event_generation ||
      readback.event_generation() != event_generation ||
      completion_frontier.event_generation() != event_generation ||
      upload.context_identity() != readback.context_identity() ||
      upload.context_identity() != completion_slot.context_identity() ||
      pinned_result_backing.size() != result_layout.total_bytes()) {
    return Status::InvalidArgument(
        "packed step transaction identity or initial state is invalid");
  }
  auto initialized = result_layout.initialize(pinned_result_backing);
  if (!initialized.ok()) return initialized;
  return QwenBf16PackedStepTransaction(
      std::move(upload), compute, std::move(readback), result_layout,
      sample_count, std::move(completion_slot), std::move(completion_frontier),
      pinned_result_backing, stream, event_generation, health_provider);
}

Status QwenBf16PackedStepTransaction::submit(
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    CompletionEventDriver& event_driver) {
  if (state_ != QwenBf16PackedStepTransactionState::kPrepared) {
    return Status::FailedPrecondition("packed step is not submit-ready");
  }
  auto status = upload_.submit(copy_driver);
  if (status.ok()) {
    status = compute_->submit(clear_driver, kernel_driver, linear_driver,
                              stream_);
  }
  if (status.ok()) status = readback_.submit(copy_driver);
  if (status.ok()) {
    status = completion_slot_.record(event_driver, stream_, event_generation_);
  }
  if (!status.ok()) {
    state_ = QwenBf16PackedStepTransactionState::kPoisoned;
    return status;
  }
  state_ = QwenBf16PackedStepTransactionState::kSubmitted;
  return Status::Ok();
}

Result<CompletionPublicationEvidence> QwenBf16PackedStepTransaction::collect() {
  auto health = health_provider_->collect();
  if (!health.ok()) return health.status();
  std::uint32_t error = UINT32_MAX;
  std::memcpy(&error,
              pinned_result_backing_.data() +
                  result_layout_.device_error().offset_bytes,
              sizeof(error));
  return CompletionPublicationEvidence{health->submit_thread_last_error_clean,
                                       error, health->engine_poisoned};
}

Result<std::vector<std::uint32_t>> QwenBf16PackedStepTransaction::poll(
    CompletionEventDriver& event_driver) {
  if (state_ != QwenBf16PackedStepTransactionState::kSubmitted) {
    return Status::FailedPrecondition("packed step is not awaiting completion");
  }
  auto status = completion_slot_.poll(event_driver, completion_frontier_, *this);
  if (!status.ok()) {
    if (status.code() != StatusCode::kUnavailable)
      state_ = QwenBf16PackedStepTransactionState::kPoisoned;
    return status;
  }
  auto tokens = result_layout_.parse(
      pinned_result_backing_, sample_count_,
      completion_frontier_.publication_authorized());
  if (!tokens.ok()) {
    state_ = QwenBf16PackedStepTransactionState::kPoisoned;
    return tokens.status();
  }
  state_ = QwenBf16PackedStepTransactionState::kCompleted;
  return tokens;
}

Result<std::vector<QwenBf16PackedSampleReceipt>>
QwenBf16PackedStepTransaction::poll_sampling(
    CompletionEventDriver& event_driver) {
  if (state_ != QwenBf16PackedStepTransactionState::kSubmitted) {
    return Status::FailedPrecondition("packed step is not awaiting completion");
  }
  auto status = completion_slot_.poll(event_driver, completion_frontier_, *this);
  if (!status.ok()) {
    if (status.code() != StatusCode::kUnavailable)
      state_ = QwenBf16PackedStepTransactionState::kPoisoned;
    return status;
  }
  auto receipts = result_layout_.parse_sampling(
      pinned_result_backing_, sample_count_,
      completion_frontier_.publication_authorized());
  if (!receipts.ok()) {
    state_ = QwenBf16PackedStepTransactionState::kPoisoned;
    return receipts.status();
  }
  state_ = QwenBf16PackedStepTransactionState::kCompleted;
  return receipts;
}

Status QwenBf16PackedStepTransaction::expire(std::uint64_t now_ns) {
  if (state_ != QwenBf16PackedStepTransactionState::kSubmitted) {
    return Status::FailedPrecondition("packed step is not awaiting completion");
  }
  auto status = completion_frontier_.expire(now_ns);
  if (status.code() != StatusCode::kUnavailable)
    state_ = QwenBf16PackedStepTransactionState::kPoisoned;
  return status;
}

Status QwenBf16PackedStepTransaction::release_completion() {
  if (state_ != QwenBf16PackedStepTransactionState::kCompleted) {
    return Status::FailedPrecondition(
        "packed completion cannot be released before publication");
  }
  return completion_slot_.release(event_generation_);
}

}  // namespace pih
