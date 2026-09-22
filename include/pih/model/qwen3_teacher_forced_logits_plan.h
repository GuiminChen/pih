#pragma once

#include "pih/model/qwen3_bf16_packed_dispatch_plan.h"
#include "pih/model/qwen3_bf16_prepared_execution.h"
#include "pih/model/qwen3_int4_prepared_execution.h"

namespace pih {

class QwenTeacherForcedLogitsPlan final {
 public:
  static Result<QwenTeacherForcedLogitsPlan> Create(
      const ResolvedKernelFunction& gather_function,
      const TensorView& normalized_rows, const TensorView& sample_row_indices,
      const TensorView& gathered_rows, const TensorView& lm_head_weight,
      const TensorView& logits, const TensorView& device_error,
      std::int32_t owning_rank);

  Status submit(KernelLaunchDriver& kernel_driver,
                QwenBf16LinearExecutionDriver& linear_driver,
                DriverStreamHandle stream);
  Status submit_int4(KernelLaunchDriver& kernel_driver,
                     QwenInt4LmHeadExecutionDriver& linear_driver,
                     DriverStreamHandle stream);
  [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] bool submitted() const noexcept { return submitted_; }
  [[nodiscard]] const TensorView& logits() const noexcept {
    return lm_head_.output();
  }

 private:
  QwenTeacherForcedLogitsPlan(QwenBf16PackedDispatchPlan gather,
                              QwenBf16LinearBinding lm_head,
                              QwenInt4LmHeadBinding int4_lm_head,
                              std::uint32_t rows)
      : gather_(std::move(gather)), lm_head_(lm_head),
        int4_lm_head_(int4_lm_head), rows_(rows) {}

  QwenBf16PackedDispatchPlan gather_;
  QwenBf16LinearBinding lm_head_;
  QwenInt4LmHeadBinding int4_lm_head_;
  std::uint32_t rows_ = 0;
  bool submitted_ = false;
};

}  // namespace pih
