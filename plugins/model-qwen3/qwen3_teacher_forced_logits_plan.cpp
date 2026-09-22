#include "pih/model/qwen3_teacher_forced_logits_plan.h"

#include <array>

namespace pih {
namespace {

bool exact_matrix(const TensorView& view, DType dtype, std::int64_t rows,
                  std::int64_t columns, std::int32_t device,
                  std::uint64_t generation) {
  return view.dtype() == dtype && view.device().type() == DeviceType::kCuda &&
         view.device().index() == device && view.generation() == generation &&
         view.rank() == 2 && view.dim(0) == rows && view.dim(1) == columns &&
         view.stride(0) == columns && view.stride(1) == 1;
}

bool overlaps(const TensorView& left, const TensorView& right) {
  auto left_bytes = dtype_size(left.dtype());
  auto right_bytes = dtype_size(right.dtype());
  if (!left_bytes.ok() || !right_bytes.ok()) return true;
  const auto left_begin = reinterpret_cast<std::uintptr_t>(left.data());
  const auto right_begin = reinterpret_cast<std::uintptr_t>(right.data());
  const auto left_end = left_begin + left.num_elements() * *left_bytes;
  const auto right_end = right_begin + right.num_elements() * *right_bytes;
  return left_begin < right_end && right_begin < left_end;
}

}  // namespace

Result<QwenTeacherForcedLogitsPlan> QwenTeacherForcedLogitsPlan::Create(
    const ResolvedKernelFunction& gather_function,
    const TensorView& normalized_rows, const TensorView& sample_row_indices,
    const TensorView& gathered_rows, const TensorView& lm_head_weight,
    const TensorView& logits, const TensorView& device_error,
    std::int32_t owning_rank) {
  if (owning_rank < 0 || normalized_rows.rank() != 2 ||
      normalized_rows.dim(0) <= 0 || normalized_rows.dim(0) > 4096 ||
      gathered_rows.rank() != 2 || gathered_rows.dim(0) <= 0 ||
      gathered_rows.dim(0) > 32) {
    return Status::InvalidArgument(
        "Qwen teacher-forced logits geometry is invalid");
  }
  const auto rows = static_cast<std::uint32_t>(gathered_rows.dim(0));
  const auto source_rows = static_cast<std::uint32_t>(normalized_rows.dim(0));
  const auto generation = normalized_rows.generation();
  if (generation == 0 ||
      !exact_matrix(normalized_rows, DType::kBFloat16, source_rows, 1024,
                    owning_rank, generation) ||
      sample_row_indices.dtype() != DType::kUInt32 ||
      sample_row_indices.device().type() != DeviceType::kCuda ||
      sample_row_indices.device().index() != owning_rank ||
      sample_row_indices.generation() != generation ||
      sample_row_indices.rank() != 1 ||
      sample_row_indices.num_elements() != rows ||
      !exact_matrix(gathered_rows, DType::kBFloat16, rows, 1024,
                    owning_rank, generation) ||
      !exact_matrix(lm_head_weight, DType::kBFloat16, 151936, 1024,
                    owning_rank, generation) ||
      !exact_matrix(logits, DType::kFloat32, rows, 151936, owning_rank,
                    generation) ||
      device_error.dtype() != DType::kUInt8 || device_error.rank() != 1 ||
      device_error.num_elements() != sizeof(std::uint32_t) ||
      device_error.device().type() != DeviceType::kCuda ||
      device_error.device().index() != owning_rank ||
      device_error.generation() != generation ||
      overlaps(gathered_rows, lm_head_weight) || overlaps(gathered_rows, logits) ||
      overlaps(lm_head_weight, logits)) {
    return Status::InvalidArgument(
        "Qwen teacher-forced logits tensor contract is invalid");
  }
  const std::array<std::int64_t, 1> wire_shape{
      static_cast<std::int64_t>(rows * sizeof(std::uint32_t))};
  auto wire_rows = TensorView::Create(
      sample_row_indices.data(), DType::kUInt8, wire_shape, {},
      sample_row_indices.device(), sample_row_indices.generation());
  if (!wire_rows.ok()) return wire_rows.status();
  auto gather = QwenBf16PackedDispatchPlan::CreateSampleHidden(
      gather_function, normalized_rows, *wire_rows, gathered_rows,
      device_error, rows, source_rows, owning_rank);
  if (!gather.ok()) return gather.status();
  QwenBf16LinearBinding lm_head(QwenBf16LinearKind::kLmHead, gathered_rows,
                                lm_head_weight, logits);
  QwenInt4LmHeadBinding int4_lm_head(gathered_rows, lm_head_weight, logits);
  return QwenTeacherForcedLogitsPlan(std::move(*gather), lm_head,
                                     int4_lm_head, rows);
}

Status QwenTeacherForcedLogitsPlan::submit(
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    DriverStreamHandle stream) {
  if (stream == 0)
    return Status::InvalidArgument(
        "Qwen teacher-forced logits requires an explicit stream");
  if (submitted_)
    return Status::FailedPrecondition(
        "Qwen teacher-forced logits plan cannot be replayed");
  submitted_ = true;
  auto status = gather_.submit(kernel_driver, stream);
  if (!status.ok()) return status;
  return linear_driver.execute(lm_head_, stream);
}

Status QwenTeacherForcedLogitsPlan::submit_int4(
    KernelLaunchDriver& kernel_driver,
    QwenInt4LmHeadExecutionDriver& linear_driver,
    DriverStreamHandle stream) {
  if (stream == 0)
    return Status::InvalidArgument(
        "Qwen teacher-forced logits requires an explicit stream");
  if (submitted_)
    return Status::FailedPrecondition(
        "Qwen teacher-forced logits plan cannot be replayed");
  submitted_ = true;
  auto status = gather_.submit(kernel_driver, stream);
  if (!status.ok()) return status;
  return linear_driver.execute(int4_lm_head_, stream);
}

}  // namespace pih
