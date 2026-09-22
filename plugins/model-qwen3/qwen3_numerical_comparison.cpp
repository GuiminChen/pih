#include "pih/model/qwen3_numerical_comparison.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pih {
namespace {

bool valid_policy(const QwenNumericalPolicy& policy) {
  return policy.max_elements != 0 &&
         std::isfinite(policy.absolute_tolerance) &&
         policy.absolute_tolerance >= 0.0 &&
         std::isfinite(policy.relative_tolerance) &&
         policy.relative_tolerance >= 0.0 &&
         std::isfinite(policy.maximum_nrmse) && policy.maximum_nrmse >= 0.0 &&
         std::isfinite(policy.minimum_cosine) &&
         policy.minimum_cosine >= -1.0 && policy.minimum_cosine <= 1.0 &&
         std::isfinite(policy.zero_rms_absolute_tolerance) &&
         policy.zero_rms_absolute_tolerance >= 0.0;
}

double nearest_rank(const std::vector<double>& sorted, double percentile) {
  const auto rank = static_cast<std::size_t>(
      std::ceil(percentile * static_cast<double>(sorted.size())));
  return sorted[std::max<std::size_t>(rank, 1) - 1];
}

}  // namespace

Result<QwenNumericalReport> compare_qwen_numerics(
    std::span<const float> reference, std::span<const float> candidate,
    const QwenNumericalPolicy& policy) {
  if (!valid_policy(policy) || reference.empty() ||
      reference.size() != candidate.size() ||
      reference.size() > policy.max_elements) {
    return Status::InvalidArgument("Qwen numerical comparison identity is invalid");
  }

  std::vector<double> absolute_errors;
  absolute_errors.reserve(reference.size());
  double reference_squared = 0.0;
  double candidate_squared = 0.0;
  double error_squared = 0.0;
  double dot = 0.0;
  std::uint64_t violations = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double expected = reference[i];
    const double actual = candidate[i];
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
      return Status::FailedPrecondition(
          "Qwen numerical comparison contains nonfinite values");
    }
    const double error = std::abs(actual - expected);
    absolute_errors.push_back(error);
    reference_squared += expected * expected;
    candidate_squared += actual * actual;
    error_squared += error * error;
    dot += expected * actual;
    if (error > policy.absolute_tolerance +
                    policy.relative_tolerance * std::abs(expected)) {
      ++violations;
    }
  }
  std::sort(absolute_errors.begin(), absolute_errors.end());
  const double count = static_cast<double>(reference.size());
  const double reference_rms = std::sqrt(reference_squared / count);
  const double error_rms = std::sqrt(error_squared / count);
  const bool zero_rms = reference_rms == 0.0;
  double cosine = 1.0;
  if (reference_squared != 0.0 && candidate_squared != 0.0) {
    cosine = dot / std::sqrt(reference_squared * candidate_squared);
    cosine = std::clamp(cosine, -1.0, 1.0);
  } else if (reference_squared != candidate_squared) {
    cosine = 0.0;
  }
  const double nrmse = zero_rms ? 0.0 : error_rms / reference_rms;
  const double maximum = absolute_errors.back();
  const bool qualified = zero_rms
                             ? maximum <= policy.zero_rms_absolute_tolerance
                             : violations == 0 &&
                                   nrmse <= policy.maximum_nrmse &&
                                   cosine >= policy.minimum_cosine;
  return QwenNumericalReport{
      static_cast<std::uint64_t>(reference.size()), violations, maximum,
      nearest_rank(absolute_errors, 0.50), nearest_rank(absolute_errors, 0.95),
      nearest_rank(absolute_errors, 0.99), reference_rms, nrmse, cosine,
      zero_rms, qualified};
}

}  // namespace pih
