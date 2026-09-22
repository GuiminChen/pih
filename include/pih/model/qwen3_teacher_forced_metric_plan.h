#pragma once

#include <cstdint>
#include <utility>

#include "pih/backend/cuda/verified_kernel_launcher.h"
#include "pih/model/qwen3_bf16_kernel_manifest.h"

namespace pih {

// Verified device-side reduction over engine-produced FP32 logits. The plan
// owns a frozen argument packet and is deliberately single-submit.
class QwenTeacherForcedMetricPlan final {
 public:
  static constexpr std::uint32_t kVocabularySize = 151936;
  static constexpr std::uint32_t kMaximumRows = 4096;

  static Result<QwenTeacherForcedMetricPlan> Create(
      const ResolvedKernelFunction& function, const TensorView& logits,
      const TensorView& target_tokens, const TensorView& argmax_tokens,
      const TensorView& target_nll, const TensorView& nonfinite_rows,
      const TensorView& error_flag, std::int32_t owning_rank);

  QwenTeacherForcedMetricPlan(const QwenTeacherForcedMetricPlan&) = delete;
  QwenTeacherForcedMetricPlan& operator=(const QwenTeacherForcedMetricPlan&) =
      delete;
  QwenTeacherForcedMetricPlan(QwenTeacherForcedMetricPlan&&) noexcept = default;
  QwenTeacherForcedMetricPlan& operator=(
      QwenTeacherForcedMetricPlan&&) noexcept = default;

  Status submit(KernelLaunchDriver& driver, DriverStreamHandle stream);
  [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] const KernelLaunchGeometry& geometry() const noexcept {
    return geometry_;
  }
  [[nodiscard]] const KernelArgumentPacket& arguments() const noexcept {
    return arguments_;
  }
  [[nodiscard]] bool submitted() const noexcept { return submitted_; }
  [[nodiscard]] const TensorView& logits() const noexcept { return logits_; }

 private:
  QwenTeacherForcedMetricPlan(ResolvedKernelFunction function,
                              KernelLaunchGeometry geometry,
                              KernelArgumentPacket arguments,
                              TensorView logits, std::uint32_t rows)
      : function_(std::move(function)), geometry_(geometry),
        arguments_(std::move(arguments)), logits_(logits), rows_(rows) {}

  ResolvedKernelFunction function_;
  KernelLaunchGeometry geometry_;
  KernelArgumentPacket arguments_;
  TensorView logits_;
  std::uint32_t rows_ = 0;
  bool submitted_ = false;
};

}  // namespace pih
