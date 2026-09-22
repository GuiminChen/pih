#include "pih/model/deepseek_rank_capacity_plan_instance.h"

#include <gtest/gtest.h>

#include <array>
#include <utility>
#include <vector>

#include "deepseek_rank_capacity_test_fixture.h"

namespace pih {
namespace {

std::vector<DeepSeekRankProcessManifest> capacity_manifests(
    std::uint32_t count) {
  std::vector<DeepSeekRankProcessManifest> result;
  for (std::uint32_t rank = 0; rank < count; ++rank) {
    result.push_back({7, 8, count, rank, 10U + rank, 20U + rank,
                      test_fixture::rank_capacity_digest(
                          static_cast<std::uint8_t>(rank + 1)),
                      static_cast<std::int32_t>(rank), 500});
  }
  return result;
}

DeepSeekRankSpawnResourcePlan capacity_plan(std::uint32_t count) {
  return {count, 3, 20, 4, 200, 10, 41};
}

TEST(DeepSeekRankCapacityPlanInstanceTest,
     CompilesProductionAuthorityForOneToFourRanks) {
  EXPECT_EQ(kDeepSeekRankCapacityPlanInstanceAbi,
            "pih_deepseek_rank_capacity_plan_instance_v1");
  EXPECT_EQ(kDeepSeekRankCapacityImplementedScopeAbi,
            "profile_device_bootstrap_rank_spawn_resource_v1");
  for (std::uint32_t count = 1; count <= 4; ++count) {
    const auto manifests = capacity_manifests(count);
    const auto plan = capacity_plan(count);
    auto admission = test_fixture::rank_capacity_admission(
        static_cast<std::uint8_t>(count));
    auto bootstrap = test_fixture::rank_capacity_bootstrap(admission);
    auto instance = DeepSeekRankCapacityPlanInstance::Compile(
        admission, bootstrap, manifests, plan, 90);
    ASSERT_TRUE(instance.ok()) << instance.status().message();
    EXPECT_EQ(instance->engine_epoch(), 7U);
    EXPECT_EQ(instance->worker_generation(), 8U);
    EXPECT_EQ(instance->world_size(), count);
    EXPECT_EQ(instance->execution_topology(),
              DeepSeekExecutionTopology::kOneProcessPerRank);
    EXPECT_EQ(instance->expected_controller_process_identity(), 90U);
    EXPECT_NE(instance->manifest_root(), Sha256Digest{});
    EXPECT_NE(instance->resource_plan_root(), Sha256Digest{});
    EXPECT_NE(instance->instance_root(), Sha256Digest{});
    EXPECT_TRUE(instance->validate_static_binding(manifests, plan).ok());
  }
}

TEST(DeepSeekRankCapacityPlanInstanceTest,
     RootBindsBootstrapControllerManifestAndResourcePlan) {
  const auto manifests = capacity_manifests(2);
  const auto plan = capacity_plan(2);
  auto first = test_fixture::rank_capacity_instance(manifests, plan, 90, 17);
  auto replay = test_fixture::rank_capacity_instance(manifests, plan, 90, 17);
  auto controller_changed = test_fixture::rank_capacity_instance(
      manifests, plan, 91, 17);
  auto generation_changed = test_fixture::rank_capacity_instance(
      manifests, plan, 90, 18);
  auto changed_manifests = manifests;
  changed_manifests[1].process_manifest_identity++;
  auto manifest_changed = test_fixture::rank_capacity_instance(
      changed_manifests, plan, 90, 17);
  auto changed_plan = plan;
  changed_plan.worker_fd_peak++;
  auto plan_changed = test_fixture::rank_capacity_instance(
      manifests, changed_plan, 90, 17);

  EXPECT_EQ(first.instance_root().hex(),
            "7823f2af9c93eaeb61ab2861e31d8e13d6a2392249b61235142c1b7486eb68d7");
  EXPECT_EQ(first.instance_root(), replay.instance_root());
  EXPECT_NE(first.instance_root(), controller_changed.instance_root());
  EXPECT_NE(first.instance_root(), generation_changed.instance_root());
  EXPECT_NE(first.instance_root(), manifest_changed.instance_root());
  EXPECT_NE(first.instance_root(), plan_changed.instance_root());
}

TEST(DeepSeekRankCapacityPlanInstanceTest,
     RejectsUnretainedOrEvidenceUnboundAdmission) {
  const auto manifests = capacity_manifests(1);
  const auto plan = capacity_plan(1);
  auto unretained = test_fixture::rank_capacity_admission(1, false, true);
  auto unretained_bootstrap =
      test_fixture::rank_capacity_bootstrap(unretained);
  EXPECT_FALSE(DeepSeekRankCapacityPlanInstance::Compile(
                   unretained, unretained_bootstrap, manifests, plan, 90)
                   .ok());

  auto unbound = test_fixture::rank_capacity_admission(1, true, false);
  auto unbound_bootstrap = test_fixture::rank_capacity_bootstrap(unbound);
  EXPECT_FALSE(DeepSeekRankCapacityPlanInstance::Compile(
                   unbound, unbound_bootstrap, manifests, plan, 90)
                   .ok());
}

TEST(DeepSeekRankCapacityPlanInstanceTest,
     RejectsBootstrapDeviceOrStaticPlanDrift) {
  const auto manifests = capacity_manifests(2);
  const auto plan = capacity_plan(2);
  auto admission = test_fixture::rank_capacity_admission(2);
  auto bootstrap = test_fixture::rank_capacity_bootstrap(admission);

  auto ordinal_drift = manifests;
  ordinal_drift[1].startup_device_ordinal = 3;
  EXPECT_FALSE(DeepSeekRankCapacityPlanInstance::Compile(
                   admission, bootstrap, ordinal_drift, plan, 90)
                   .ok());
  auto count_drift = plan;
  count_drift.worker_processes = 1;
  EXPECT_FALSE(DeepSeekRankCapacityPlanInstance::Compile(
                   admission, bootstrap, manifests, count_drift, 90)
                   .ok());
  EXPECT_FALSE(DeepSeekRankCapacityPlanInstance::Compile(
                   admission, bootstrap, manifests, plan, 0)
                   .ok());

  auto wrong_fds = RuntimeProfileInheritedFdManifest::Create(
      {3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14}).value();
  std::array<std::byte, kRuntimeReadinessNonceBytes> nonce{};
  nonce.fill(std::byte{0x33});
  auto wrong_bootstrap = RuntimeProfileSupervisorBootstrapManifest::Create(
      std::move(wrong_fds), 17, test_fixture::rank_capacity_digest(99),
      test_fixture::rank_capacity_digest(60), nonce).value();
  EXPECT_FALSE(DeepSeekRankCapacityPlanInstance::Compile(
                   admission, wrong_bootstrap, manifests, plan, 90)
                   .ok());
}

TEST(DeepSeekRankCapacityPlanInstanceTest,
     MoveConsumesTheOnlyUsableInstanceValue) {
  const auto manifests = capacity_manifests(1);
  const auto plan = capacity_plan(1);
  auto source = test_fixture::rank_capacity_instance(manifests, plan);
  const auto expected_root = source.instance_root();
  auto destination = std::move(source);

  EXPECT_EQ(destination.instance_root(), expected_root);
  EXPECT_EQ(source.instance_root(), Sha256Digest{});
  EXPECT_FALSE(source.validate_static_binding(manifests, plan).ok());
}

}  // namespace
}  // namespace pih
