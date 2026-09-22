#include "pih/model/qwen3_numerical_comparison.h"

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

QwenNumericalPolicy policy() {
  return {8, 0.01, 0.10, 0.11, 0.99, 0.001};
}

TEST(QwenNumericalComparisonTest, ReportsHandDerivedContinuousMetrics) {
  const std::vector<float> reference{1.0F, 2.0F};
  const std::vector<float> candidate{1.1F, 1.8F};
  auto result = compare_qwen_numerics(reference, candidate, policy());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->finite_count, 2);
  EXPECT_NEAR(result->max_absolute_error, 0.2, 1e-6);
  EXPECT_NEAR(result->p50_absolute_error, 0.1, 1e-6);
  EXPECT_NEAR(result->p95_absolute_error, 0.2, 1e-6);
  EXPECT_NEAR(result->p99_absolute_error, 0.2, 1e-6);
  EXPECT_NEAR(result->reference_rms, std::sqrt(2.5), 1e-6);
  EXPECT_NEAR(result->nrmse, 0.1, 1e-6);
  EXPECT_NEAR(result->cosine_similarity,
              4.7 / (std::sqrt(5.0) * std::sqrt(4.45)), 1e-6);
  EXPECT_EQ(result->elementwise_violations, 0);
  EXPECT_TRUE(result->qualified);
  EXPECT_FALSE(result->used_zero_rms_gate);
}

TEST(QwenNumericalComparisonTest, ThresholdFailureReturnsReportNotError) {
  const std::vector<float> reference{1.0F, 2.0F};
  const std::vector<float> candidate{1.5F, 1.0F};
  auto result = compare_qwen_numerics(reference, candidate, policy());
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result->qualified);
  EXPECT_GT(result->elementwise_violations, 0);
  EXPECT_GT(result->nrmse, 0.11);
}

TEST(QwenNumericalComparisonTest, ZeroRmsUsesIndependentAbsoluteGate) {
  const std::vector<float> reference{0.0F, 0.0F};
  const std::vector<float> within{0.0005F, -0.0009F};
  auto accepted = compare_qwen_numerics(reference, within, policy());
  ASSERT_TRUE(accepted.ok());
  EXPECT_TRUE(accepted->used_zero_rms_gate);
  EXPECT_TRUE(accepted->qualified);
  EXPECT_DOUBLE_EQ(accepted->nrmse, 0.0);

  const std::vector<float> outside{0.0F, 0.002F};
  auto rejected = compare_qwen_numerics(reference, outside, policy());
  ASSERT_TRUE(rejected.ok());
  EXPECT_TRUE(rejected->used_zero_rms_gate);
  EXPECT_FALSE(rejected->qualified);
}

TEST(QwenNumericalComparisonTest, RejectsUnboundedMalformedAndNonfiniteInput) {
  const std::vector<float> one{1.0F};
  const std::vector<float> two{1.0F, 2.0F};
  const std::vector<float> finite_pair{1.0F, 2.0F};
  const std::vector<float> nonfinite{
      std::numeric_limits<float>::infinity()};
  EXPECT_FALSE(compare_qwen_numerics({}, {}, policy()).ok());
  EXPECT_FALSE(compare_qwen_numerics(one, two, policy()).ok());
  auto tiny_budget = policy();
  tiny_budget.max_elements = 1;
  EXPECT_FALSE(compare_qwen_numerics(finite_pair, finite_pair, tiny_budget)
                   .ok());
  EXPECT_FALSE(compare_qwen_numerics(nonfinite, one, policy()).ok());
  auto malformed = policy();
  malformed.minimum_cosine = 1.1;
  EXPECT_FALSE(compare_qwen_numerics(one, one, malformed).ok());
}

}  // namespace
}  // namespace pih
