#include "pih/model/qwen3_teacher_forced_chunk_transaction.h"

#include <cstring>

namespace pih {
namespace {

bool same_tensor(const TensorView& left, const TensorView& right) {
  if (left.data() != right.data() || left.dtype() != right.dtype() ||
      left.device() != right.device() || left.generation() != right.generation() ||
      left.rank() != right.rank())
    return false;
  for (std::size_t axis = 0; axis < left.rank(); ++axis)
    if (left.dim(axis) != right.dim(axis) ||
        left.stride(axis) != right.stride(axis))
      return false;
  return true;
}

}  // namespace

Result<QwenTeacherForcedChunkTransaction>
QwenTeacherForcedChunkTransaction::Create(
    QwenTeacherForcedMetricTransfer transfer,
    QwenTeacherForcedLogitsPlan logits_plan,
    QwenTeacherForcedMetricPlan metric_plan, TensorView device_error,
    QwenTeacherForcedMetricResultLayout result_layout,
    CompletionEventSlot completion_slot,
    CudaCompletionFrontier completion_frontier,
    std::span<std::byte> pinned_result, DriverStreamHandle stream,
    std::uint64_t event_generation, std::int32_t owning_rank,
    QwenBf16StepHealthProvider& health_provider,
    QwenTeacherForcedMetricArenaLease arena_lease) {
  if (stream == 0 || event_generation == 0 || owning_rank < 0 ||
      transfer.state() != QwenTeacherForcedMetricTransferState::kPrepared ||
      logits_plan.submitted() || metric_plan.submitted() ||
      logits_plan.rows() == 0 || logits_plan.rows() != metric_plan.rows() ||
      !same_tensor(logits_plan.logits(), metric_plan.logits()) ||
      metric_plan.rows() > result_layout.row_capacity() ||
      transfer.stream() != stream ||
      transfer.event_generation() != event_generation ||
      completion_slot.state() != CompletionEventSlotState::kIdle ||
      completion_slot.context_identity() != transfer.context_identity() ||
      completion_frontier.event_generation() != event_generation ||
      pinned_result.size() != result_layout.total_bytes() ||
      device_error.dtype() != DType::kUInt32 || device_error.rank() != 1 ||
      device_error.dim(0) != 1 ||
      device_error.device().type() != DeviceType::kCuda ||
      device_error.device().index() != owning_rank ||
      device_error.generation() == 0) {
    return Status::InvalidArgument(
        "Qwen teacher-forced chunk transaction identity is invalid");
  }
  auto status = result_layout.initialize(pinned_result);
  if (!status.ok()) return status;
  return QwenTeacherForcedChunkTransaction(
      std::move(transfer), std::move(logits_plan), std::move(metric_plan),
      device_error, result_layout, std::move(completion_slot),
      std::move(completion_frontier), pinned_result, stream, event_generation,
      owning_rank, health_provider, std::move(arena_lease));
}

Status QwenTeacherForcedChunkTransaction::submit_bf16(
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& head_driver,
    CompletionEventDriver& event_driver) {
  if (state_ != QwenTeacherForcedChunkTransactionState::kPrepared)
    return Status::FailedPrecondition(
        "Qwen teacher-forced chunk transaction cannot submit");
  auto status = arena_lease_.mark_submitted();
  if (status.ok()) status = transfer_.submit_upload(copy_driver);
  if (status.ok())
    status = clear_driver.clear_u32_async(device_error_, owning_rank_, stream_);
  if (status.ok())
    status = logits_plan_.submit(kernel_driver, head_driver, stream_);
  if (status.ok()) status = metric_plan_.submit(kernel_driver, stream_);
  if (status.ok()) status = transfer_.submit_readback(copy_driver);
  if (status.ok())
    status = completion_slot_.record(event_driver, stream_, event_generation_);
  if (!status.ok()) {
    state_ = QwenTeacherForcedChunkTransactionState::kPoisoned;
    return status;
  }
  submitted_role_ = QwenTeacherForcedChunkRole::kBf16;
  state_ = QwenTeacherForcedChunkTransactionState::kSubmitted;
  return Status::Ok();
}

Status QwenTeacherForcedChunkTransaction::submit_int4(
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenInt4LmHeadExecutionDriver& head_driver,
    CompletionEventDriver& event_driver) {
  if (state_ != QwenTeacherForcedChunkTransactionState::kPrepared)
    return Status::FailedPrecondition(
        "Qwen teacher-forced chunk transaction cannot submit");
  auto status = arena_lease_.mark_submitted();
  if (status.ok()) status = transfer_.submit_upload(copy_driver);
  if (status.ok())
    status = clear_driver.clear_u32_async(device_error_, owning_rank_, stream_);
  if (status.ok())
    status = logits_plan_.submit_int4(kernel_driver, head_driver, stream_);
  if (status.ok()) status = metric_plan_.submit(kernel_driver, stream_);
  if (status.ok()) status = transfer_.submit_readback(copy_driver);
  if (status.ok())
    status = completion_slot_.record(event_driver, stream_, event_generation_);
  if (!status.ok()) {
    state_ = QwenTeacherForcedChunkTransactionState::kPoisoned;
    return status;
  }
  submitted_role_ = QwenTeacherForcedChunkRole::kInt4;
  state_ = QwenTeacherForcedChunkTransactionState::kSubmitted;
  return Status::Ok();
}

Result<CompletionPublicationEvidence>
QwenTeacherForcedChunkTransaction::collect() {
  auto health = health_provider_->collect();
  if (!health.ok()) return health.status();
  std::uint32_t error = UINT32_MAX;
  std::memcpy(&error, pinned_result_.data() +
                          result_layout_.device_error().offset_bytes,
              sizeof(error));
  return CompletionPublicationEvidence{
      health->submit_thread_last_error_clean, error, health->engine_poisoned};
}

Result<QwenTeacherForcedMetricBatch>
QwenTeacherForcedChunkTransaction::poll(
    CompletionEventDriver& event_driver) {
  if (state_ != QwenTeacherForcedChunkTransactionState::kSubmitted)
    return Status::FailedPrecondition(
        "Qwen teacher-forced chunk transaction is not pending");
  auto status = completion_slot_.poll(event_driver, completion_frontier_, *this);
  if (!status.ok()) {
    if (status.code() != StatusCode::kUnavailable)
      state_ = QwenTeacherForcedChunkTransactionState::kPoisoned;
    return status;
  }
  auto result = result_layout_.parse(
      pinned_result_, metric_plan_.rows(),
      completion_frontier_.publication_authorized());
  if (!result.ok()) {
    state_ = QwenTeacherForcedChunkTransactionState::kPoisoned;
    return result.status();
  }
  state_ = QwenTeacherForcedChunkTransactionState::kCompleted;
  return result;
}

Status QwenTeacherForcedChunkTransaction::expire(std::uint64_t now_ns) {
  if (state_ != QwenTeacherForcedChunkTransactionState::kSubmitted)
    return Status::FailedPrecondition(
        "Qwen teacher-forced chunk transaction is not pending");
  auto status = completion_frontier_.expire(now_ns);
  if (status.code() != StatusCode::kUnavailable)
    state_ = QwenTeacherForcedChunkTransactionState::kPoisoned;
  return status;
}

Status QwenTeacherForcedChunkTransaction::release_completion() {
  if (state_ != QwenTeacherForcedChunkTransactionState::kCompleted)
    return Status::FailedPrecondition(
        "Qwen teacher-forced chunk completion is not releasable");
  auto status = completion_slot_.release(event_generation_);
  if (!status.ok()) return status;
  status = arena_lease_.release_completed();
  if (!status.ok()) {
    state_ = QwenTeacherForcedChunkTransactionState::kPoisoned;
    return status;
  }
  return Status::Ok();
}

}  // namespace pih
