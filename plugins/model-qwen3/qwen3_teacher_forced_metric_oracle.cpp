#include "pih/model/qwen3_teacher_forced_metric_oracle.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace pih {

Result<QwenTeacherForcedMetricBatch> qwen_teacher_forced_metric_oracle(
    std::span<const float> logits, std::span<const std::uint32_t> target_tokens,
    std::uint32_t vocabulary_size) {
  if (target_tokens.empty() || vocabulary_size == 0 ||
      target_tokens.size() >
          std::numeric_limits<std::size_t>::max() / vocabulary_size ||
      logits.size() != target_tokens.size() * vocabulary_size) {
    return Status::InvalidArgument(
        "Qwen teacher-forced metric shape is invalid");
  }
  QwenTeacherForcedMetricBatch result;
  result.rows.reserve(target_tokens.size());
  double aggregate = 0.0;
  double aggregate_correction = 0.0;
  for (std::size_t row = 0; row < target_tokens.size(); ++row) {
    const auto target = target_tokens[row];
    if (target >= vocabulary_size) {
      return Status::InvalidArgument(
          "Qwen teacher-forced target token is out of range");
    }
    const auto values = logits.subspan(row * vocabulary_size, vocabulary_size);
    bool finite = true;
    std::uint32_t argmax = 0;
    float maximum = values[0];
    for (std::uint32_t token = 0; token < vocabulary_size; ++token) {
      if (!std::isfinite(values[token])) finite = false;
      if (values[token] > maximum) {
        maximum = values[token];
        argmax = token;
      }
    }
    if (!finite) {
      ++result.nonfinite_count;
      result.rows.push_back({argmax, 0.0, false});
      continue;
    }
    double exponential_sum = 0.0;
    double correction = 0.0;
    for (const float value : values) {
      const double term = std::exp(static_cast<double>(value - maximum));
      const double adjusted = term - correction;
      const double next = exponential_sum + adjusted;
      correction = (next - exponential_sum) - adjusted;
      exponential_sum = next;
    }
    const double nll = std::log(exponential_sum) +
                       static_cast<double>(maximum) -
                       static_cast<double>(values[target]);
    const double adjusted = nll - aggregate_correction;
    const double next = aggregate + adjusted;
    aggregate_correction = (next - aggregate) - adjusted;
    aggregate = next;
    result.rows.push_back({argmax, nll, true});
  }
  result.nll_sum = aggregate;
  return result;
}

}  // namespace pih
