#include "pih/model/qwen3_teacher_forced_metric_result_layout.h"

#include <cmath>
#include <cstring>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_cuda_invariant.h"

namespace pih {

Result<QwenTeacherForcedMetricResultLayout>
QwenTeacherForcedMetricResultLayout::Create(std::uint32_t row_capacity) {
  if (row_capacity == 0 || row_capacity > kMaximumRows)
    return Status::InvalidArgument("Qwen metric result capacity is invalid");
  auto u32_bytes = checked_mul_u64(row_capacity, sizeof(std::uint32_t));
  auto f64_bytes = checked_mul_u64(row_capacity, sizeof(double));
  if (!u32_bytes.ok()) return u32_bytes.status();
  if (!f64_bytes.ok()) return f64_bytes.status();
  auto nll_offset = checked_align_up_u64(*u32_bytes, kAlignment);
  if (!nll_offset.ok()) return nll_offset.status();
  auto nll_end = checked_add_u64(*nll_offset, *f64_bytes);
  if (!nll_end.ok()) return nll_end.status();
  auto nonfinite_offset = checked_align_up_u64(*nll_end, kAlignment);
  if (!nonfinite_offset.ok()) return nonfinite_offset.status();
  auto nonfinite_end = checked_add_u64(*nonfinite_offset, *u32_bytes);
  if (!nonfinite_end.ok()) return nonfinite_end.status();
  auto error_offset = checked_align_up_u64(*nonfinite_end, kAlignment);
  if (!error_offset.ok()) return error_offset.status();
  auto error_end = checked_add_u64(*error_offset, sizeof(std::uint32_t));
  if (!error_end.ok()) return error_end.status();
  auto total = checked_align_up_u64(*error_end, kAlignment);
  if (!total.ok()) return total.status();
  return QwenTeacherForcedMetricResultLayout(
      row_capacity, {0, *u32_bytes}, {*nll_offset, *f64_bytes},
      {*nonfinite_offset, *u32_bytes},
      {*error_offset, sizeof(std::uint32_t)}, *total);
}

Status QwenTeacherForcedMetricResultLayout::initialize(
    std::span<std::byte> backing) const {
  if (backing.size() != total_bytes_)
    return Status::InvalidArgument("Qwen metric result backing is invalid");
  std::memset(backing.data(), 0xff, backing.size());
  return Status::Ok();
}

Result<QwenTeacherForcedMetricBatch>
QwenTeacherForcedMetricResultLayout::parse(
    std::span<const std::byte> backing, std::uint32_t rows,
    bool publication_authorized) const {
  if (!publication_authorized || backing.size() != total_bytes_ || rows == 0 ||
      rows > row_capacity_)
    return Status::FailedPrecondition(
        "Qwen metric result is not authorized for publication");
  std::uint32_t error = UINT32_MAX;
  std::memcpy(&error, backing.data() + device_error_.offset_bytes,
              sizeof(error));
  if (!qwen_cuda_invariant_known(error) ||
      error != static_cast<std::uint32_t>(QwenCudaInvariant::kNone))
    return Status::Internal("Qwen metric device invariant rejected publication");
  QwenTeacherForcedMetricBatch result;
  result.rows.reserve(rows);
  double sum = 0.0, correction = 0.0;
  for (std::uint32_t row = 0; row < rows; ++row) {
    std::uint32_t argmax = UINT32_MAX, nonfinite = UINT32_MAX;
    double nll = 0.0;
    std::memcpy(&argmax, backing.data() + argmax_tokens_.offset_bytes +
                             row * sizeof(argmax), sizeof(argmax));
    std::memcpy(&nll, backing.data() + target_nll_.offset_bytes +
                         row * sizeof(nll), sizeof(nll));
    std::memcpy(&nonfinite, backing.data() + nonfinite_rows_.offset_bytes +
                               row * sizeof(nonfinite), sizeof(nonfinite));
    if (argmax >= kVocabularySize || nonfinite > 1U ||
        (nonfinite == 0U && (!std::isfinite(nll) || nll < 0.0)) ||
        (nonfinite == 1U && nll != 0.0))
      return Status::Internal("Qwen metric row receipt is invalid");
    if (nonfinite != 0U) {
      ++result.nonfinite_count;
      result.rows.push_back({argmax, 0.0, false});
      continue;
    }
    const double adjusted = nll - correction;
    const double next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
    result.rows.push_back({argmax, nll, true});
  }
  result.nll_sum = sum;
  return result;
}

}  // namespace pih
