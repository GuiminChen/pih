#include "pih/model/qwen3_teacher_forced_metric_oracle.h"

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace pih {

TEST(QwenTeacherForcedMetricOracle, UsesStableLogSumExpAndLowestTieId) {
  const std::vector<float> logits{
      1000.0F, 1000.0F, 999.0F,
      -1000.0F, -999.0F, -998.0F,
  };
  const std::vector<std::uint32_t> targets{1, 2};
  auto result = qwen_teacher_forced_metric_oracle(logits, targets, 3);
  ASSERT_TRUE(result.ok()) << result.status().message();
  ASSERT_EQ(result->rows.size(), 2U);
  EXPECT_EQ(result->rows[0].argmax_token, 0U);
  EXPECT_EQ(result->rows[1].argmax_token, 2U);
  EXPECT_TRUE(result->rows[0].finite);
  EXPECT_NEAR(result->rows[0].target_nll,
              std::log(2.0 + std::exp(-1.0)), 1e-12);
  EXPECT_NEAR(result->nll_sum,
              result->rows[0].target_nll + result->rows[1].target_nll,
              1e-12);
}

TEST(QwenTeacherForcedMetricOracle, CountsNonfiniteRowsWithoutHidingThem) {
  const std::vector<float> logits{
      1.0F, std::numeric_limits<float>::infinity(), 0.0F,
      1.0F, 2.0F, 3.0F,
  };
  const std::vector<std::uint32_t> targets{0, 2};
  auto result = qwen_teacher_forced_metric_oracle(logits, targets, 3);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->nonfinite_count, 1U);
  EXPECT_FALSE(result->rows[0].finite);
  EXPECT_TRUE(result->rows[1].finite);
  EXPECT_DOUBLE_EQ(result->nll_sum, result->rows[1].target_nll);
}

TEST(QwenTeacherForcedMetricOracle, RejectsShapeAndTargetDrift) {
  const std::vector<float> logits{1.0F, 2.0F};
  const std::vector<std::uint32_t> target{0};
  EXPECT_FALSE(qwen_teacher_forced_metric_oracle(logits, target, 3).ok());
  const std::vector<std::uint32_t> invalid{2};
  EXPECT_FALSE(qwen_teacher_forced_metric_oracle(logits, invalid, 2).ok());
}

}  // namespace pih
