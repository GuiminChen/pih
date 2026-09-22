#include "pih/model/deepseek_learned_router_coordinator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace pih { namespace {
class RouterOperations final : public DeepSeekLearnedRouterOperations {
 public:
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    ++zeros; return Status::Ok();
  }
  Status gemm(DeepSeekRouterBf16GemmLaunch launch) override {
    product = launch; return Status::Ok();
  }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override {
    ++copies; return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    ++records; return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return polls++ == 0 ? DeepSeekExpertAsyncStatus::kInProgress
                        : DeepSeekExpertAsyncStatus::kSuccess;
  }
  int zeros = 0, copies = 0, records = 0, polls = 0;
  DeepSeekRouterBf16GemmLaunch product{};
};
struct RouterFixture final {
  DeepSeekPipelinePlanDescriptor descriptor{1, 1, DeepSeekPlanPhase::kDecode,
                                             1, 1};
  DeepSeekRouteScratchArena scratch =
      DeepSeekRouteScratchArena::Create(1).value();
  DeepSeekExpertPlanCatalog catalog =
      DeepSeekExpertPlanCatalog::Create(descriptor, {3, 3}).value();
  RouterOperations operations;
  std::array<float, 256> bias{};
  DeepSeekLearnedRouterCoordinator coordinator =
      DeepSeekLearnedRouterCoordinator::Create(
          1, bias, scratch, catalog, operations).value();
  std::vector<float> scores = [] {
    std::vector<float> result(256);
    for (std::uint32_t i = 0; i < result.size(); ++i)
      result[i] = static_cast<float>(i) / 16.0F;
    return result;
  }();
  std::uint32_t host_error = 0;
  DeepSeekLearnedRouterSubmission submission() {
    return {3, 1, 1, 2, 3, 4, 5, 6, scores, &host_error};
  }
};
TEST(DeepSeekLearnedRouterCoordinatorTest,
     PreservesFp32ProjectionUntilRouteCatalogPublication) {
  RouterFixture fixture;
  ASSERT_TRUE(fixture.coordinator.launch(fixture.submission()).ok());
  EXPECT_EQ(fixture.operations.zeros, 1);
  EXPECT_EQ(fixture.operations.copies, 2);
  EXPECT_EQ(fixture.operations.records, 1);
  EXPECT_EQ(fixture.operations.product.expert_count, 256U);
  EXPECT_EQ(fixture.operations.product.hidden_size, 4096U);
  EXPECT_EQ(fixture.coordinator.poll().value(),
            DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(fixture.coordinator.poll().value(),
            DeepSeekStageComputeStatus::kSuccess);
  auto plan = fixture.catalog.resolve(3, fixture.descriptor);
  ASSERT_TRUE(plan.ok());
  const auto first = std::find_if(
      (*plan)->routes().begin(), (*plan)->routes().end(),
      [](const auto& route) { return route.route_ordinal == 0; });
  const auto sixth = std::find_if(
      (*plan)->routes().begin(), (*plan)->routes().end(),
      [](const auto& route) { return route.route_ordinal == 5; });
  ASSERT_NE(first, (*plan)->routes().end());
  ASSERT_NE(sixth, (*plan)->routes().end());
  EXPECT_EQ(first->expert_id, 255U);
  EXPECT_EQ(sixth->expert_id, 250U);
}
TEST(DeepSeekLearnedRouterCoordinatorTest,
     KernelErrorProducesNoRoutePlanAndPoisonsCoordinator) {
  RouterFixture fixture;
  ASSERT_TRUE(fixture.coordinator.launch(fixture.submission()).ok());
  ASSERT_EQ(fixture.coordinator.poll().value(),
            DeepSeekStageComputeStatus::kInProgress);
  fixture.host_error = 4;
  EXPECT_EQ(fixture.coordinator.poll().value(),
            DeepSeekStageComputeStatus::kError);
  EXPECT_FALSE(fixture.catalog.resolve(3, fixture.descriptor).ok());
  EXPECT_FALSE(fixture.coordinator.launch(fixture.submission()).ok());
}
TEST(DeepSeekLearnedRouterCoordinatorTest,
     RejectsHashLayersAndMismatchedHostScoreExtent) {
  RouterFixture fixture;
  auto submission = fixture.submission(); submission.layer = 2;
  EXPECT_FALSE(fixture.coordinator.launch(submission).ok());
  submission = fixture.submission(); submission.host_scores = {};
  EXPECT_FALSE(fixture.coordinator.launch(submission).ok());
}
}}  // namespace pih::<anonymous>
