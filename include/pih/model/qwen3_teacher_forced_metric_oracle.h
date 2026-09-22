#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

struct QwenTeacherForcedMetricRow final {
  std::uint32_t argmax_token = 0;
  double target_nll = 0.0;
  bool finite = false;
};

struct QwenTeacherForcedMetricBatch final {
  std::vector<QwenTeacherForcedMetricRow> rows;
  double nll_sum = 0.0;
  std::uint64_t nonfinite_count = 0;
};

// Independent qualification oracle for the future CUDA row reducer. Ties use
// the smallest token id and NLL uses stable log-sum-exp over raw FP32 logits.
Result<QwenTeacherForcedMetricBatch> qwen_teacher_forced_metric_oracle(
    std::span<const float> logits, std::span<const std::uint32_t> target_tokens,
    std::uint32_t vocabulary_size);

}  // namespace pih
