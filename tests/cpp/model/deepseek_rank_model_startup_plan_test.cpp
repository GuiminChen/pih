#include "pih/model/deepseek_rank_model_startup_plan.h"

#include "deepseek_rank_capacity_test_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace pih {
namespace {

Sha256Digest digest(std::uint8_t seed) {
  Sha256Digest value{};
  value.bytes.fill(static_cast<std::byte>(seed));
  return value;
}

std::vector<DeepSeekRankProcessManifest> manifests(std::uint32_t count) {
  std::vector<DeepSeekRankProcessManifest> result;
  for (std::uint32_t rank = 0; rank < count; ++rank) {
    result.push_back({7, 8, count, rank, 10U + rank, 20U + rank,
                      digest(static_cast<std::uint8_t>(rank + 1)),
                      static_cast<std::int32_t>(rank), 500});
  }
  return result;
}

DeepSeekRankSpawnResourcePlan spawn_plan(std::uint32_t count) {
  return {count, 3, 20, 4, 200, 10, 41};
}

DeepSeekRankSpawnResourceObservation spawn_observation() {
  return {10,
          100,
          true,
          {{digest(50), 5, 100}, {digest(51), 10, 200}},
          10,
          100,
          200,
          1000,
          100,
          1000,
          1000};
}

DeepSeekRankPostExecResourcePlan resource_plan() {
  return {8, 20, 200, 2, 4, 10, digest(30)};
}

class ProcessDriver final : public DeepSeekRankProcessDriver {
 public:
  Result<DeepSeekRankProcessHandle> spawn(
      const DeepSeekRankProcessManifest& manifest) override {
    return DeepSeekRankProcessHandle{100U + manifest.rank,
                                     200U + manifest.rank,
                                     300U + manifest.rank};
  }
  Result<DeepSeekRankProcessObservation> observe(
      const DeepSeekRankProcessHandle&) override {
    return DeepSeekRankProcessObservation::kRunning;
  }
  Status terminate(const DeepSeekRankProcessHandle&) override {
    return Status::Ok();
  }
  Status send_challenge(
      const DeepSeekRankProcessHandle&,
      const DeepSeekRankExecChallenge& challenge) override {
    challenges.push_back(challenge);
    return Status::Ok();
  }
  Result<std::optional<DeepSeekRankExecReady>> poll_ready(
      const DeepSeekRankProcessHandle& handle) override {
    for (const auto& challenge : challenges) {
      if (challenge.handle.process_identity != handle.process_identity) {
        continue;
      }
      const auto& manifest = challenge.manifest;
      return std::optional<DeepSeekRankExecReady>{{
          {manifest.engine_epoch, manifest.worker_generation, manifest.rank,
           manifest.physical_device_identity,
           manifest.process_manifest_identity, handle.process_identity,
           handle.pidfd_identity, handle.control_identity,
           manifest.physical_device_uuid_commitment,
           manifest.startup_device_ordinal, manifest.startup_deadline_ns},
          challenge.challenge_identity}};
    }
    return std::optional<DeepSeekRankExecReady>{};
  }
  std::vector<DeepSeekRankExecChallenge> challenges;
};

class StartupFixture final {
 public:
  static std::unique_ptr<StartupFixture> Create(std::uint32_t count,
                                                std::uint64_t deployment = 17) {
    auto result = std::unique_ptr<StartupFixture>(new StartupFixture);
    result->manifests_ = manifests(count);
    result->spawn_ = spawn_plan(count);
    result->admission_.emplace(test_fixture::rank_capacity_admission(
        static_cast<std::uint8_t>(count)));
    result->bootstrap_.emplace(test_fixture::rank_capacity_bootstrap(
        *result->admission_, deployment));
    auto capacity_instance = DeepSeekRankCapacityPlanInstance::Compile(
        *result->admission_, *result->bootstrap_, result->manifests_,
        result->spawn_, 90).value();
    auto preflight = DeepSeekRankSpawnPreflightReceipt::Compile(
        result->manifests_, result->spawn_, spawn_observation()).value();
    auto authorization = DeepSeekRankSpawnAuthorization::Create(
        result->manifests_, std::move(capacity_instance), preflight).value();
    result->supervisor_.emplace(
        DeepSeekRankProcessSupervisor::Create(
            result->manifests_, std::move(authorization), result->driver_)
            .value());
    EXPECT_TRUE(result->supervisor_->launch().ok());
    std::vector<std::uint64_t> challenges;
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      challenges.push_back(400U + rank);
    }
    EXPECT_TRUE(result->supervisor_->dispatch_challenges(90, challenges).ok());
    EXPECT_TRUE(result->supervisor_->advance_exec_startup(499).ok());

    result->resource_plans_.assign(count, resource_plan());
    std::vector<DeepSeekRankPostExecResourceReceipt> receipts;
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      const auto& manifest = result->manifests_[rank];
      DeepSeekRankPostExecResourceObservation observation{
          manifest.engine_epoch,
          manifest.worker_generation,
          rank,
          100U + rank,
          400U + rank,
          result->supervisor_->capacity_plan_instance_root(),
          digest(30),
          8,
          20,
          0,
          200,
          24,
          100,
          1000,
          210,
          true};
      receipts.push_back(DeepSeekRankPostExecResourceReceipt::Compile(
                             result->manifests_, result->spawn_,
                             result->resource_plans_[rank],
                             *result->supervisor_->exec_ready(rank), observation)
                             .value());
    }
    result->seal_.emplace(DeepSeekRankPostExecResourceSeal::Compile(
                              *result->supervisor_, result->manifests_,
                              result->spawn_, result->resource_plans_, receipts)
                              .value());
    std::array<std::byte, kRuntimeReadinessNonceBytes> nonce{};
    nonce.fill(std::byte{0x33});
    result->readiness_.emplace(issue_runtime_profile_readiness_receipt(
        *result->admission_, *result->bootstrap_, nonce,
        result->bootstrap_->authority_snapshot_root(),
        result->supervisor_->capacity_plan_instance_root()).value());
    return result;
  }

  Result<DeepSeekRankModelStartupPlan> compile(
      const DeepSeekPipelinePlan& pipeline,
      const DeepSeekPipelineCapacity& capacity,
      std::uint64_t deadline = 900) const {
    return DeepSeekRankModelStartupPlan::Compile(
        *bootstrap_, *readiness_, *supervisor_, manifests_, *seal_, pipeline,
        capacity, deadline);
  }

  const RuntimeEngineAdmission& admission() const { return *admission_; }
  const RuntimeProfileReadinessReceipt& readiness() const {
    return *readiness_;
  }
  const RuntimeProfileSupervisorBootstrapManifest& bootstrap() const {
    return *bootstrap_;
  }
  const DeepSeekRankProcessSupervisor& supervisor() const {
    return *supervisor_;
  }
  const DeepSeekRankPostExecResourceSeal& seal() const { return *seal_; }
  const std::vector<DeepSeekRankProcessManifest>& process_manifests() const {
    return manifests_;
  }

 private:
  StartupFixture() = default;
  std::vector<DeepSeekRankProcessManifest> manifests_;
  DeepSeekRankSpawnResourcePlan spawn_;
  ProcessDriver driver_;
  std::optional<RuntimeEngineAdmission> admission_;
  std::optional<RuntimeProfileSupervisorBootstrapManifest> bootstrap_;
  std::optional<DeepSeekRankProcessSupervisor> supervisor_;
  std::vector<DeepSeekRankPostExecResourcePlan> resource_plans_;
  std::optional<DeepSeekRankPostExecResourceSeal> seal_;
  std::optional<RuntimeProfileReadinessReceipt> readiness_;
};

TEST(DeepSeekRankModelStartupPlanTest,
     CompilesExactOneToFourRankGenerationBindings) {
  EXPECT_EQ(kDeepSeekRankModelStartupPlanAbi,
            "pih_deepseek_rank_model_startup_plan_v1");
  for (std::uint32_t count = 1; count <= 4; ++count) {
    auto fixture = StartupFixture::Create(count);
    auto pipeline = DeepSeekPipelinePlan::Create(count, false);
    auto capacity = DeepSeekPipelineCapacity::Create(count, 17, 9, 3, false);
    ASSERT_TRUE(pipeline.ok());
    ASSERT_TRUE(capacity.ok());
    auto plan = fixture->compile(*pipeline, *capacity);
    ASSERT_TRUE(plan.ok()) << plan.status().message();
    EXPECT_EQ(plan->engine_epoch(), 7U);
    EXPECT_EQ(plan->worker_generation(), 8U);
    EXPECT_EQ(plan->world_size(), count);
    EXPECT_EQ(plan->model_startup_deadline_ns(), 900U);
    EXPECT_EQ(plan->manifest_root(), fixture->supervisor().manifest_root());
    EXPECT_EQ(plan->capacity_plan_instance_root(),
              fixture->supervisor().capacity_plan_instance_root());
    EXPECT_EQ(plan->post_exec_resource_seal_root(),
              fixture->seal().seal_root());
    EXPECT_TRUE(plan->admission_authority_retained());
    EXPECT_NE(plan->plan_root(), Sha256Digest{});
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      EXPECT_NE(plan->rank_seed_root(rank), Sha256Digest{});
      const auto& seed = plan->rank_seed(rank);
      EXPECT_EQ(seed.rank, rank);
      EXPECT_EQ(seed.process_manifest_identity, 20U + rank);
      EXPECT_EQ(seed.process_identity, 100U + rank);
      EXPECT_EQ(seed.pidfd_identity, 200U + rank);
      EXPECT_EQ(seed.control_identity, 300U + rank);
      EXPECT_EQ(seed.challenge_identity, 400U + rank);
      EXPECT_EQ(seed.physical_device_identity, 10U + rank);
      EXPECT_EQ(seed.startup_device_ordinal,
                static_cast<std::int32_t>(rank));
      EXPECT_EQ(seed.physical_device_uuid_commitment,
                digest(static_cast<std::uint8_t>(rank + 1)));
      EXPECT_EQ(seed.seed_root, plan->rank_seed_root(rank));
      if (rank != 0) {
        EXPECT_NE(plan->rank_seed_root(rank), plan->rank_seed_root(0));
      }
    }
    EXPECT_THROW((void)plan->rank_seed_root(count), std::out_of_range);
    EXPECT_THROW((void)plan->rank_seed(count), std::out_of_range);
  }
}

TEST(DeepSeekRankModelStartupPlanTest,
     RootBindsCapacityPipelineDeadlineAndProfileReadiness) {
  auto fixture = StartupFixture::Create(2);
  auto later_fixture = StartupFixture::Create(2, 18);
  auto pipeline = DeepSeekPipelinePlan::Create(2, false).value();
  auto capacity = DeepSeekPipelineCapacity::Create(2, 17, 9, 3, false).value();
  auto first = fixture->compile(pipeline, capacity).value();
  auto replay = fixture->compile(pipeline, capacity).value();
  auto larger = DeepSeekPipelineCapacity::Create(2, 18, 9, 3, false).value();
  auto changed_capacity = fixture->compile(pipeline, larger).value();
  auto changed_deadline = fixture->compile(pipeline, capacity, 901).value();
  auto changed_readiness = later_fixture->compile(pipeline, capacity).value();

  EXPECT_EQ(first.plan_root(), replay.plan_root());
  EXPECT_NE(first.plan_root(), changed_capacity.plan_root());
  EXPECT_NE(first.plan_root(), changed_deadline.plan_root());
  EXPECT_NE(first.plan_root(), changed_readiness.plan_root());
}

TEST(DeepSeekRankModelStartupPlanTest,
     RejectsWorldDsparkDeadlineAndProfileSplices) {
  auto fixture = StartupFixture::Create(2);
  auto pipeline = DeepSeekPipelinePlan::Create(2, false).value();
  auto capacity = DeepSeekPipelineCapacity::Create(2, 17, 9, 3, false).value();
  EXPECT_FALSE(fixture->compile(pipeline, capacity, 500).ok());

  auto wrong_world_pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  EXPECT_FALSE(fixture->compile(wrong_world_pipeline, capacity).ok());
  auto wrong_world_capacity =
      DeepSeekPipelineCapacity::Create(1, 17, 9, 3, false).value();
  EXPECT_FALSE(fixture->compile(pipeline, wrong_world_capacity).ok());
  auto dspark_capacity =
      DeepSeekPipelineCapacity::Create(2, 17, 9, 3, true).value();
  EXPECT_FALSE(fixture->compile(pipeline, dspark_capacity).ok());

  auto foreign_fixture = StartupFixture::Create(1);
  EXPECT_FALSE(DeepSeekRankModelStartupPlan::Compile(
                   fixture->bootstrap(), foreign_fixture->readiness(),
                   fixture->supervisor(),
                   fixture->process_manifests(),
                   fixture->seal(), pipeline, capacity, 900)
                   .ok());
  auto foreign_bootstrap = StartupFixture::Create(2, 18);
  EXPECT_FALSE(DeepSeekRankModelStartupPlan::Compile(
                   foreign_bootstrap->bootstrap(), fixture->readiness(),
                   fixture->supervisor(), fixture->process_manifests(),
                   fixture->seal(), pipeline, capacity, 900)
                   .ok());
}

}  // namespace
}  // namespace pih
