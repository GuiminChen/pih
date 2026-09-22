#include "pih/model/deepseek_bound_router_stage_work_provider.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "pih/model/deepseek_bound_expert_plan_provider.h"

namespace pih {
namespace {

class Ops final : public DeepSeekLearnedRouterOperations {
 public:
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status gemm(DeepSeekRouterBf16GemmLaunch) override { return Status::Ok(); }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

TEST(DeepSeekBoundRouterStageWorkProviderTest,
     AtomicallyBindsEveryHashLayerAndInjectsCanonicalStore) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({0, 1}, 4);
  auto provider = DeepSeekBoundRouterStageWorkProvider::Create({0, 1}, *store);
  auto scratch0 = DeepSeekRouteScratchArena::Create(4);
  auto scratch1 = DeepSeekRouteScratchArena::Create(4);
  ASSERT_TRUE(store.ok()); ASSERT_TRUE(provider.ok());
  ASSERT_TRUE(scratch0.ok()); ASSERT_TRUE(scratch1.ok());
  std::vector<std::uint32_t> tokens{0};
  std::vector<float> scores(256);
  std::vector<std::uint16_t> table{0,1,2,3,4,5};
  const std::array<DeepSeekBoundHashRouterWork, 2> work{{
      {1, tokens, scores, 1, table, &*scratch1},
      {0, tokens, scores, 1, table, &*scratch0}}};
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  ASSERT_TRUE(provider->bind(descriptor, work, {}).ok());

  auto resolved = provider->resolve_hash(
      {DeepSeekStageOperatorKind::kMoe, 0}, descriptor);
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(resolved->store, static_cast<DeepSeekExpertPlanStore*>(&*store));
  EXPECT_EQ(resolved->scratch, &*scratch0);
  auto stale = descriptor;
  ++stale.plan_sequence;
  EXPECT_FALSE(provider->resolve_hash(
      {DeepSeekStageOperatorKind::kMoe, 0}, stale).ok());
  EXPECT_FALSE(provider->resolve_hash(
      {DeepSeekStageOperatorKind::kAttention, 0}, descriptor).ok());
}

TEST(DeepSeekBoundRouterStageWorkProviderTest,
     FailedIncompleteRebindPreservesPriorPlan) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({0, 1}, 4);
  auto provider = DeepSeekBoundRouterStageWorkProvider::Create({0, 1}, *store);
  auto scratch = DeepSeekRouteScratchArena::Create(4);
  ASSERT_TRUE(store.ok()); ASSERT_TRUE(provider.ok()); ASSERT_TRUE(scratch.ok());
  std::vector<std::uint32_t> tokens{0};
  std::vector<float> scores(256);
  std::vector<std::uint16_t> table{0,1,2,3,4,5};
  const std::array<DeepSeekBoundHashRouterWork, 2> complete{{
      {0, tokens, scores, 1, table, &*scratch},
      {1, tokens, scores, 1, table, &*scratch}}};
  const DeepSeekPipelinePlanDescriptor first{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  ASSERT_TRUE(provider->bind(first, complete, {}).ok());
  auto second = first;
  ++second.plan_sequence;
  EXPECT_FALSE(provider->bind(second,
      std::span<const DeepSeekBoundHashRouterWork>(complete.data(), 1), {}).ok());
  EXPECT_TRUE(provider->resolve_hash(
      {DeepSeekStageOperatorKind::kMoe, 1}, first).ok());
  EXPECT_FALSE(provider->resolve_hash(
      {DeepSeekStageOperatorKind::kMoe, 1}, second).ok());
}

TEST(DeepSeekBoundRouterStageWorkProviderTest,
     RejectsLearnedCoordinatorUsingForeignStore) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({3, 3}, 4);
  auto foreign = DeepSeekBoundExpertPlanProvider::Create({3, 3}, 4);
  auto provider = DeepSeekBoundRouterStageWorkProvider::Create({3, 3}, *store);
  auto scratch = DeepSeekRouteScratchArena::Create(4);
  ASSERT_TRUE(store.ok()); ASSERT_TRUE(foreign.ok()); ASSERT_TRUE(provider.ok());
  ASSERT_TRUE(scratch.ok());
  Ops ops;
  std::array<float, 256> bias{};
  auto coordinator = DeepSeekLearnedRouterCoordinator::Create(
      4, bias, *scratch, *foreign, ops);
  ASSERT_TRUE(coordinator.ok());
  std::vector<float> host_scores(256);
  std::uint32_t error = 0;
  const DeepSeekLearnedRouterSubmission submission{
      3, 1, 1, 2, 3, 4, 5, 6, host_scores, &error};
  const std::array<DeepSeekBoundLearnedRouterWork, 1> work{{
      {3, &*coordinator, submission}}};
  EXPECT_FALSE(provider->bind(
      {7, 9, DeepSeekPlanPhase::kDecode, 1, 1}, {}, work).ok());
}

TEST(DeepSeekBoundRouterStageWorkProviderTest,
     RejectsSyntheticDsparkLayerIdentity) {
  auto store = DeepSeekBoundExpertPlanProvider::Create({42, 42}, 4).value();
  EXPECT_FALSE(DeepSeekBoundRouterStageWorkProvider::Create(
      {43, 43}, store).ok());
}

}  // namespace
}  // namespace pih
