#include "pih/model/qwen3_teacher_forced_metric_transfer.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenTeacherForcedMetricTransfer>
QwenTeacherForcedMetricTransfer::Create(
    const QwenTeacherForcedMetricResultLayout& layout, std::uint32_t rows,
    CudaCopyEndpoint pinned_sample_rows,
    CudaCopyEndpoint device_sample_rows,
    CudaCopyEndpoint pinned_targets, CudaCopyEndpoint device_targets,
    CudaCopyEndpoint device_result, CudaCopyEndpoint pinned_result,
    std::uintptr_t context_identity, DriverStreamHandle stream,
    std::uint64_t event_generation, std::uint64_t row_upload_plan_id,
    std::uint64_t target_upload_plan_id,
    std::uint64_t readback_plan_id) {
  if (rows == 0 || rows > layout.row_capacity() || row_upload_plan_id == 0 ||
      target_upload_plan_id == 0 || readback_plan_id == 0 ||
      row_upload_plan_id == target_upload_plan_id ||
      row_upload_plan_id == readback_plan_id ||
      target_upload_plan_id == readback_plan_id)
    return Status::InvalidArgument("Qwen metric transfer identity is invalid");
  auto target_bytes = checked_mul_u64(rows, sizeof(std::uint32_t));
  if (!target_bytes.ok()) return target_bytes.status();
  auto row_upload = CudaTypedCopyPlan::Create(
      row_upload_plan_id, CudaCopyPurpose::kInput,
      CudaCopyKind::kHostToDevice, pinned_sample_rows, device_sample_rows,
      *target_bytes, 4, context_identity, stream, event_generation);
  if (!row_upload.ok()) return row_upload.status();
  auto target_upload = CudaTypedCopyPlan::Create(
      target_upload_plan_id, CudaCopyPurpose::kInput,
      CudaCopyKind::kHostToDevice,
      pinned_targets, device_targets, *target_bytes, 4, context_identity,
      stream, event_generation);
  if (!target_upload.ok()) return target_upload.status();
  auto readback = CudaTypedCopyPlan::Create(
      readback_plan_id, CudaCopyPurpose::kResult,
      CudaCopyKind::kDeviceToHost, device_result, pinned_result,
      layout.total_bytes(), QwenTeacherForcedMetricResultLayout::kAlignment,
      context_identity, stream, event_generation);
  if (!readback.ok()) return readback.status();
  return QwenTeacherForcedMetricTransfer(std::move(*row_upload),
                                         std::move(*target_upload),
                                         std::move(*readback));
}

Status QwenTeacherForcedMetricTransfer::submit_upload(
    TypedCopyDriver& driver) {
  if (state_ != QwenTeacherForcedMetricTransferState::kPrepared)
    return Status::FailedPrecondition("Qwen metric upload cannot replay");
  auto status = row_upload_.submit(driver);
  if (status.ok()) status = target_upload_.submit(driver);
  if (!status.ok()) {
    state_ = QwenTeacherForcedMetricTransferState::kPoisoned;
    return status;
  }
  state_ = QwenTeacherForcedMetricTransferState::kUploaded;
  return Status::Ok();
}

Status QwenTeacherForcedMetricTransfer::submit_readback(
    TypedCopyDriver& driver) {
  if (state_ != QwenTeacherForcedMetricTransferState::kUploaded)
    return Status::FailedPrecondition(
        "Qwen metric readback requires completed upload submission");
  auto status = readback_.submit(driver);
  if (!status.ok()) {
    state_ = QwenTeacherForcedMetricTransferState::kPoisoned;
    return status;
  }
  state_ = QwenTeacherForcedMetricTransferState::kReadbackSubmitted;
  return Status::Ok();
}

}  // namespace pih
