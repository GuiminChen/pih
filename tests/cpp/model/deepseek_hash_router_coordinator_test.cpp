#include "pih/model/deepseek_hash_router_coordinator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace pih { namespace {

class HashProjectionOperations final : public DeepSeekLearnedRouterOperations {
 public:
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    ++zeros;
    return Status::Ok();
  }
  Status gemm(DeepSeekRouterBf16GemmLaunch launch) override {
    product = launch;
    return Status::Ok();
  }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override {
    ++copies;
    return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    ++records;
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return polls++ == 0 ? DeepSeekExpertAsyncStatus::kInProgress
                        : DeepSeekExpertAsyncStatus::kSuccess;
  }
  int zeros = 0, copies = 0, records = 0, polls = 0;
  DeepSeekRouterBf16GemmLaunch product{};
};

struct HashProjectionFixture final {
  DeepSeekPipelinePlanDescriptor descriptor{
      1, 1, DeepSeekPlanPhase::kDecode, 1, 1};
  DeepSeekRouteScratchArena scratch =
      DeepSeekRouteScratchArena::Create(1).value();
  DeepSeekExpertPlanCatalog catalog =
      DeepSeekExpertPlanCatalog::Create(descriptor, {0, 0}).value();
  HashProjectionOperations operations;
  DeepSeekHashRouterCoordinator coordinator =
      DeepSeekHashRouterCoordinator::Create(
          1, scratch, catalog, operations).value();
  std::vector<std::uint32_t> tokens{1};
  std::vector<std::uint16_t> table{
      0, 1, 2, 3, 4, 5,
      10, 11, 12, 13, 14, 15};
  std::vector<float> scores = [] {
    std::vector<float> result(256, -10.0F);
    for (std::uint32_t expert = 10; expert <= 15; ++expert) {
      result[expert] = static_cast<float>(expert);
    }
    return result;
  }();
  std::uint32_t host_error = 0;

  DeepSeekHashRouterSubmission submission() {
    return {{0, 1, 1, 2, 3, 4, 5, 6, scores, &host_error},
            tokens, 2, table};
  }
};

TEST(DeepSeekHashRouterCoordinatorTest,
     ProjectsScoresBeforeTokenTableRoutePublication) {
  HashProjectionFixture fixture;
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
  auto plan = fixture.catalog.resolve(0, fixture.descriptor);
  ASSERT_TRUE(plan.ok());
  const auto first = std::find_if(
      (*plan)->routes().begin(), (*plan)->routes().end(),
      [](const auto& route) { return route.route_ordinal == 0; });
  ASSERT_NE(first, (*plan)->routes().end());
  EXPECT_EQ(first->expert_id, 10U);
}

TEST(DeepSeekHashRouterCoordinatorTest,
     RejectsLearnedLayerAndPublishesNothingAfterGpuError) {
  HashProjectionFixture fixture;
  auto learned = fixture.submission();
  learned.projection.layer = 3;
  EXPECT_FALSE(fixture.coordinator.launch(learned).ok());
  ASSERT_TRUE(fixture.coordinator.launch(fixture.submission()).ok());
  EXPECT_EQ(fixture.coordinator.poll().value(),
            DeepSeekStageComputeStatus::kInProgress);
  fixture.host_error = 1;
  EXPECT_EQ(fixture.coordinator.poll().value(),
            DeepSeekStageComputeStatus::kError);
  EXPECT_FALSE(fixture.catalog.resolve(0, fixture.descriptor).ok());
}

}}  // namespace pih::<anonymous>
