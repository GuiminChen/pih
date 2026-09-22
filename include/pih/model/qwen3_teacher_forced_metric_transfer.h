#pragma once

#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/model/qwen3_teacher_forced_metric_result_layout.h"

namespace pih {

enum class QwenTeacherForcedMetricTransferState : std::uint8_t {
  kPrepared, kUploaded, kReadbackSubmitted, kPoisoned,
};

class QwenTeacherForcedMetricTransfer final {
 public:
  static Result<QwenTeacherForcedMetricTransfer> Create(
      const QwenTeacherForcedMetricResultLayout& layout, std::uint32_t rows,
      CudaCopyEndpoint pinned_sample_rows,
      CudaCopyEndpoint device_sample_rows,
      CudaCopyEndpoint pinned_targets, CudaCopyEndpoint device_targets,
      CudaCopyEndpoint device_result, CudaCopyEndpoint pinned_result,
      std::uintptr_t context_identity, DriverStreamHandle stream,
      std::uint64_t event_generation, std::uint64_t row_upload_plan_id,
      std::uint64_t target_upload_plan_id,
      std::uint64_t readback_plan_id);

  Status submit_upload(TypedCopyDriver& driver);
  Status submit_readback(TypedCopyDriver& driver);
  [[nodiscard]] QwenTeacherForcedMetricTransferState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uintptr_t context_identity() const noexcept {
    return row_upload_.context_identity();
  }
  [[nodiscard]] DriverStreamHandle stream() const noexcept {
    return row_upload_.stream();
  }
  [[nodiscard]] std::uint64_t event_generation() const noexcept {
    return row_upload_.completion_event_generation();
  }

 private:
  QwenTeacherForcedMetricTransfer(CudaTypedCopyPlan row_upload,
                                  CudaTypedCopyPlan target_upload,
                                  CudaTypedCopyPlan readback)
      : row_upload_(std::move(row_upload)),
        target_upload_(std::move(target_upload)),
        readback_(std::move(readback)) {}
  CudaTypedCopyPlan row_upload_;
  CudaTypedCopyPlan target_upload_;
  CudaTypedCopyPlan readback_;
  QwenTeacherForcedMetricTransferState state_ =
      QwenTeacherForcedMetricTransferState::kPrepared;
};

}  // namespace pih
