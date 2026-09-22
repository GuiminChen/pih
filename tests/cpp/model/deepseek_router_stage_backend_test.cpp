#include "pih/model/deepseek_router_stage_backend.h"

#include <array>
#include <vector>

#include <gtest/gtest.h>

namespace pih { namespace {

DeepSeekPipelinePlanDescriptor plan() {
  return {7, 13, DeepSeekPlanPhase::kDecode, 1, 1};
}

class Inner final : public DeepSeekStageOperatorBackend {
 public:
  explicit Inner(DeepSeekExpertPlanCatalog& catalog) : catalog_(&catalog) {}
  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& descriptor) override {
    if (command.kind == DeepSeekStageOperatorKind::kMoe) {
      auto resolved = catalog_->resolve(command.layer, descriptor);
      if (!resolved.ok()) return resolved.status();
    }
    launches.push_back(command);
    return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    return DeepSeekStageComputeStatus::kSuccess;
  }
  DeepSeekExpertPlanCatalog* catalog_;
  std::vector<DeepSeekStageOperatorCommand> launches;
};

class Ops final : public DeepSeekLearnedRouterOperations {
 public:
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status gemm(DeepSeekRouterBf16GemmLaunch launch) override {
    EXPECT_EQ(launch.expert_count, 256U);
    return Status::Ok();
  }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return ready ? DeepSeekExpertAsyncStatus::kSuccess
                 : DeepSeekExpertAsyncStatus::kInProgress;
  }
  bool ready = false;
};

class Provider final : public DeepSeekRouterStageWorkProvider {
 public:
  DeepSeekExpertPlanProvider* plan_provider() noexcept override {
    return provider;
  }
  Result<DeepSeekHashRouterStageWork> resolve_hash(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override { return hash; }
  Result<DeepSeekLearnedRouterStageWork> resolve_learned(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override { return learned; }
  DeepSeekHashRouterStageWork hash;
  DeepSeekLearnedRouterStageWork learned;
  DeepSeekExpertPlanProvider* provider = nullptr;
};

TEST(DeepSeekRouterStageBackendTest, PublishesHashPlanBeforeExpertBackendLaunch) {
  auto catalog = DeepSeekExpertPlanCatalog::Create(plan(), {0, 0});
  auto scratch = DeepSeekRouteScratchArena::Create(1);
  ASSERT_TRUE(catalog.ok()); ASSERT_TRUE(scratch.ok());
  std::vector<std::uint32_t> tokens{1};
  std::vector<float> scores(256, 0.0F);
  std::vector<std::uint16_t> table(12);
  for (std::uint16_t i = 0; i < 6; ++i) {
    table[i] = i; table[6 + i] = static_cast<std::uint16_t>(i + 6);
  }
  Provider provider;
  provider.provider = &*catalog;
  provider.hash = {tokens, scores, 2, table, &*scratch, &*catalog};
  Inner inner(*catalog);
  auto backend = DeepSeekRouterStageOperatorBackend::Create(inner, provider);
  ASSERT_TRUE(backend.ok());
  ASSERT_TRUE(backend->launch({DeepSeekStageOperatorKind::kMoe, 0}, plan()).ok());
  ASSERT_EQ(inner.launches.size(), 1U);
  auto done = backend->poll(); ASSERT_TRUE(done.ok());
  EXPECT_EQ(*done, DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekRouterStageBackendTest, WaitsForLearnedPlanBeforeExpertBackendLaunch) {
  auto catalog = DeepSeekExpertPlanCatalog::Create(plan(), {3, 3});
  auto scratch = DeepSeekRouteScratchArena::Create(1);
  ASSERT_TRUE(catalog.ok()); ASSERT_TRUE(scratch.ok());
  Ops ops; std::array<float, 256> bias{};
  auto coordinator = DeepSeekLearnedRouterCoordinator::Create(
      1, bias, *scratch, *catalog, ops);
  ASSERT_TRUE(coordinator.ok());
  std::vector<float> host_scores(256);
  for (std::uint32_t i = 0; i < 256; ++i) host_scores[i] = static_cast<float>(i);
  std::uint32_t error = 0;
  Provider provider;
  provider.provider = &*catalog;
  provider.learned = {&*coordinator,
      {3, 1, 1, 2, 3, 4, 5, 6, host_scores, &error}};
  Inner inner(*catalog);
  auto backend = DeepSeekRouterStageOperatorBackend::Create(inner, provider);
  ASSERT_TRUE(backend.ok());
  ASSERT_TRUE(backend->launch({DeepSeekStageOperatorKind::kMoe, 3}, plan()).ok());
  EXPECT_TRUE(inner.launches.empty());
  auto waiting = backend->poll(); ASSERT_TRUE(waiting.ok());
  EXPECT_EQ(*waiting, DeepSeekStageComputeStatus::kInProgress);
  EXPECT_TRUE(inner.launches.empty());
  ops.ready = true;
  auto routed = backend->poll(); ASSERT_TRUE(routed.ok());
  EXPECT_EQ(*routed, DeepSeekStageComputeStatus::kInProgress);
  ASSERT_EQ(inner.launches.size(), 1U);
  auto done = backend->poll(); ASSERT_TRUE(done.ok());
  EXPECT_EQ(*done, DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekRouterStageBackendTest,
     WaitsForProjectedHashPlanBeforeExpertBackendLaunch) {
  auto catalog = DeepSeekExpertPlanCatalog::Create(plan(), {0, 0});
  auto scratch = DeepSeekRouteScratchArena::Create(1);
  ASSERT_TRUE(catalog.ok()); ASSERT_TRUE(scratch.ok());
  Ops ops;
  auto coordinator = DeepSeekHashRouterCoordinator::Create(
      1, *scratch, *catalog, ops);
  ASSERT_TRUE(coordinator.ok());
  std::vector<std::uint32_t> tokens{0};
  std::vector<float> scores(256);
  std::vector<std::uint16_t> table{0, 1, 2, 3, 4, 5};
  std::uint32_t error = 0;
  Provider provider;
  provider.provider = &*catalog;
  provider.hash.coordinator = &*coordinator;
  provider.hash.submission = {
      {0, 1, 1, 2, 3, 4, 5, 6, scores, &error}, tokens, 1, table};
  Inner inner(*catalog);
  auto backend = DeepSeekRouterStageOperatorBackend::Create(inner, provider);
  ASSERT_TRUE(backend.ok());
  ASSERT_TRUE(backend->launch(
      {DeepSeekStageOperatorKind::kMoe, 0}, plan()).ok());
  EXPECT_TRUE(inner.launches.empty());
  EXPECT_EQ(backend->poll().value(), DeepSeekStageComputeStatus::kInProgress);
  ops.ready = true;
  EXPECT_EQ(backend->poll().value(), DeepSeekStageComputeStatus::kInProgress);
  ASSERT_EQ(inner.launches.size(), 1U);
  EXPECT_EQ(backend->poll().value(), DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekRouterStageBackendTest, DuplicateHashPublicationPoisonsBeforeExpertLaunch) {
  auto catalog = DeepSeekExpertPlanCatalog::Create(plan(), {0, 0});
  auto scratch = DeepSeekRouteScratchArena::Create(1);
  ASSERT_TRUE(catalog.ok()); ASSERT_TRUE(scratch.ok());
  std::vector<std::uint32_t> tokens{0};
  std::vector<float> scores(256);
  std::vector<std::uint16_t> table{0,1,2,3,4,5};
  ASSERT_TRUE(DeepSeekHashRouter::RouteInto(
      0, tokens, scores, 1, table, *scratch, *catalog).ok());
  Provider provider;
  provider.provider = &*catalog;
  provider.hash = {tokens, scores, 1, table, &*scratch, &*catalog};
  Inner inner(*catalog);
  auto backend = DeepSeekRouterStageOperatorBackend::Create(inner, provider);
  ASSERT_TRUE(backend.ok());
  EXPECT_FALSE(backend->launch({DeepSeekStageOperatorKind::kMoe, 0}, plan()).ok());
  EXPECT_TRUE(inner.launches.empty());
  EXPECT_FALSE(backend->poll().ok());
}

TEST(DeepSeekRouterStageBackendTest,
     RejectsHashWorkPublishingIntoDifferentPlanProvider) {
  auto declared = DeepSeekExpertPlanCatalog::Create(plan(), {0, 0});
  auto foreign = DeepSeekExpertPlanCatalog::Create(plan(), {0, 0});
  auto scratch = DeepSeekRouteScratchArena::Create(1);
  ASSERT_TRUE(declared.ok()); ASSERT_TRUE(foreign.ok()); ASSERT_TRUE(scratch.ok());
  std::vector<std::uint32_t> tokens{0};
  std::vector<float> scores(256);
  std::vector<std::uint16_t> table{0,1,2,3,4,5};
  Provider provider;
  provider.provider = &*declared;
  provider.hash = {tokens, scores, 1, table, &*scratch, &*foreign};
  Inner inner(*declared);
  auto backend = DeepSeekRouterStageOperatorBackend::Create(inner, provider);
  ASSERT_TRUE(backend.ok());
  EXPECT_FALSE(backend->launch(
      {DeepSeekStageOperatorKind::kMoe, 0}, plan()).ok());
  EXPECT_TRUE(inner.launches.empty());
}

TEST(DeepSeekRouterStageBackendTest,
     RejectsLearnedCoordinatorPublishingIntoDifferentPlanProvider) {
  auto declared = DeepSeekExpertPlanCatalog::Create(plan(), {3, 3});
  auto foreign = DeepSeekExpertPlanCatalog::Create(plan(), {3, 3});
  auto scratch = DeepSeekRouteScratchArena::Create(1);
  ASSERT_TRUE(declared.ok()); ASSERT_TRUE(foreign.ok()); ASSERT_TRUE(scratch.ok());
  Ops ops;
  std::array<float, 256> bias{};
  auto coordinator = DeepSeekLearnedRouterCoordinator::Create(
      1, bias, *scratch, *foreign, ops);
  ASSERT_TRUE(coordinator.ok());
  std::vector<float> host_scores(256);
  std::uint32_t error = 0;
  Provider provider;
  provider.provider = &*declared;
  provider.learned = {&*coordinator,
      {3, 1, 1, 2, 3, 4, 5, 6, host_scores, &error}};
  Inner inner(*declared);
  auto backend = DeepSeekRouterStageOperatorBackend::Create(inner, provider);
  ASSERT_TRUE(backend.ok());
  EXPECT_FALSE(backend->launch(
      {DeepSeekStageOperatorKind::kMoe, 3}, plan()).ok());
  EXPECT_TRUE(inner.launches.empty());
}

} }  // namespace pih
