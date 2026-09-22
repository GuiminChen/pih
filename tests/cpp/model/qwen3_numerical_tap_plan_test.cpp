#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_numerical_tap_plan.h"

namespace pih {
namespace {

TEST(QwenNumericalTapPlanTest, FreezesAlignedTypedSnapshotLayout) {
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 0, 3},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kQueryAfterNorm, 13, 17, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kKvValue, 27, 129, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 129, 1},
  };

  auto plan = QwenNumericalTapPlan::Create(requests, 1ULL << 20);

  ASSERT_TRUE(plan.ok()) << plan.status().message();
  ASSERT_EQ(plan->captures().size(), 4);
  EXPECT_EQ(plan->captures()[0].dtype, DType::kBFloat16);
  EXPECT_EQ(plan->captures()[0].element_count, 3 * 1024);
  EXPECT_EQ(plan->captures()[0].size_bytes, 3 * 1024 * 2);
  EXPECT_EQ(plan->captures()[1].element_count, 16 * 128);
  EXPECT_EQ(plan->captures()[2].element_count, 8 * 128);
  EXPECT_EQ(plan->captures()[3].dtype, DType::kFloat32);
  EXPECT_EQ(plan->captures()[3].element_count, 151936);
  for (const auto& capture : plan->captures()) {
    EXPECT_EQ(capture.offset_bytes % QwenNumericalTapPlan::kAlignment, 0);
    EXPECT_LE(capture.offset_bytes + capture.size_bytes, plan->arena_bytes());
  }
  EXPECT_EQ(plan->arena_bytes() % QwenNumericalTapPlan::kAlignment, 0);
}

TEST(QwenNumericalTapPlanTest, RejectsDuplicateOrSemanticallyInvalidTap) {
  const QwenNumericalTapRequest hidden{
      QwenNumericalTapPoint::kLayerHidden, 0, 0, 1};
  const std::array duplicate{hidden, hidden};
  EXPECT_EQ(QwenNumericalTapPlan::Create(duplicate, 1ULL << 20)
                .status()
                .code(),
            StatusCode::kInvalidArgument);

  const std::array bad_layer{QwenNumericalTapRequest{
      QwenNumericalTapPoint::kQueryAfterRope, 28, 0, 1}};
  EXPECT_FALSE(QwenNumericalTapPlan::Create(bad_layer, 1ULL << 20).ok());

  const std::array bad_logits{QwenNumericalTapRequest{
      QwenNumericalTapPoint::kLogits, 27, 0, 2}};
  EXPECT_FALSE(QwenNumericalTapPlan::Create(bad_logits, 1ULL << 20).ok());

  const std::array bad_position{QwenNumericalTapRequest{
      QwenNumericalTapPoint::kKvKey, 0, 40960, 1}};
  EXPECT_FALSE(QwenNumericalTapPlan::Create(bad_position, 1ULL << 20).ok());
}

TEST(QwenNumericalTapPlanTest, EnforcesEntryAndPreallocatedArenaBounds) {
  const std::array request{QwenNumericalTapRequest{
      QwenNumericalTapPoint::kLogits, 28, 0, 1}};
  auto too_small = QwenNumericalTapPlan::Create(request, 1024);
  ASSERT_FALSE(too_small.ok());
  EXPECT_EQ(too_small.status().code(), StatusCode::kResourceExhausted);

  std::array<QwenNumericalTapRequest,
             QwenNumericalTapPlan::kMaximumCaptureCount + 1>
      too_many{};
  for (std::size_t index = 0; index < too_many.size(); ++index) {
    too_many[index] = {QwenNumericalTapPoint::kLayerHidden, 0,
                       static_cast<std::uint32_t>(index), 1};
  }
  auto bounded = QwenNumericalTapPlan::Create(too_many, 1ULL << 30);
  ASSERT_FALSE(bounded.ok());
  EXPECT_EQ(bounded.status().code(), StatusCode::kResourceExhausted);
}

std::vector<QwenNumericalTapRequest> complete_baseline() {
  std::vector<QwenNumericalTapRequest> requests;
  for (const std::uint32_t layer : {0U, 13U, 27U}) {
    requests.push_back({QwenNumericalTapPoint::kLayerHidden, layer, 0, 1});
  }
  for (std::uint32_t layer = 0; layer < 28; ++layer) {
    for (const auto point : {QwenNumericalTapPoint::kQueryAfterNorm,
                             QwenNumericalTapPoint::kKeyAfterNorm,
                             QwenNumericalTapPoint::kQueryAfterRope,
                             QwenNumericalTapPoint::kKeyAfterRope,
                             QwenNumericalTapPoint::kPrefillAttention}) {
      requests.push_back({point, layer, 0, 1});
    }
    for (const std::uint32_t position : {2U, 17U, 129U, 4097U}) {
      requests.push_back(
          {QwenNumericalTapPoint::kKvKey, layer, position, 1});
      requests.push_back(
          {QwenNumericalTapPoint::kKvValue, layer, position, 1});
    }
  }
  requests.push_back({QwenNumericalTapPoint::kFinalNorm, 28, 0, 1});
  requests.push_back({QwenNumericalTapPoint::kLogits, 28, 0, 1});
  return requests;
}

TEST(QwenNumericalTapPlanTest, RequiresCompleteBf16BaselineCoverage) {
  auto requests = complete_baseline();
  auto plan = QwenNumericalTapPlan::Create(requests, 1ULL << 30);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_TRUE(validate_qwen_bf16_tap_coverage(*plan).ok());

  requests.erase(requests.begin() + 100);
  auto incomplete = QwenNumericalTapPlan::Create(requests, 1ULL << 30);
  ASSERT_TRUE(incomplete.ok());
  auto status = validate_qwen_bf16_tap_coverage(*incomplete);
  EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
