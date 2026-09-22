#pragma once

#include <cstdint>
#include <span>

#include "pih/core/result.h"

namespace pih {

struct QwenNumericalPolicy final {
  std::uint64_t max_elements;
  double absolute_tolerance;
  double relative_tolerance;
  double maximum_nrmse;
  double minimum_cosine;
  double zero_rms_absolute_tolerance;
};

struct QwenNumericalReport final {
  std::uint64_t finite_count;
  std::uint64_t elementwise_violations;
  double max_absolute_error;
  double p50_absolute_error;
  double p95_absolute_error;
  double p99_absolute_error;
  double reference_rms;
  double nrmse;
  double cosine_similarity;
  bool used_zero_rms_gate;
  bool qualified;
};

Result<QwenNumericalReport> compare_qwen_numerics(
    std::span<const float> reference, std::span<const float> candidate,
    const QwenNumericalPolicy& policy);

}  // namespace pih
