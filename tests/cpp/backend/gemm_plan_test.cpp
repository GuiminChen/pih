#include <cstdint>

#include <gtest/gtest.h>

#include "pih/backend/cuda/gemm_plan.h"

namespace pih {
namespace {

TEST(GemmPlanTest, RejectsZeroAndUnaddressableDimensions) {
  EXPECT_FALSE(GemmPlan::Create(0, 16, 16, 0).ok());
  EXPECT_FALSE(GemmPlan::Create(16, 0, 16, 0).ok());
  EXPECT_FALSE(GemmPlan::Create(16, 16, 0, 0).ok());
  EXPECT_FALSE(GemmPlan::Create(1ULL << 40, 16, 16, 0).ok());
  EXPECT_FALSE(GemmPlan::Create(16, 16, 16, 0, DType::kFloat16).ok());
}

TEST(GemmPlanTest, FreezesAnAlgorithmAtCreation) {
  auto plan = GemmPlan::Create(16, 16, 16, 1 << 20);
  if (!plan.ok()) {
    GTEST_SKIP() << plan.status().message();
  }
  EXPECT_GE(plan->get()->algorithm_id(), 0);
  EXPECT_LE(plan->get()->workspace_bytes(), 1 << 20);
}

TEST(GemmPlanTest, FreezesFp32OutputForLogitPlans) {
  auto plan = GemmPlan::Create(1, 151936, 1024, 1 << 20, DType::kFloat32);
  if (!plan.ok()) {
    GTEST_SKIP() << plan.status().message();
  }
  EXPECT_EQ(plan->get()->output_dtype(), DType::kFloat32);
}

}  // namespace
}  // namespace pih
