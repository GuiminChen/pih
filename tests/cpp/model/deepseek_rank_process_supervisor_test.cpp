#include "pih/model/deepseek_rank_process_supervisor.h"
#include "deepseek_rank_capacity_test_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <utility>

namespace pih {
namespace {

Sha256Digest device_commitment(std::uint32_t rank) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(rank + 1));
  return result;
}

class ProcessDriver final : public DeepSeekRankProcessDriver {
 public:
  Result<DeepSeekRankProcessHandle> spawn(
      const DeepSeekRankProcessManifest& manifest) override {
    spawned.push_back(manifest.rank);
    if (duplicate_handles && manifest.rank != 0) {
      return DeepSeekRankProcessHandle{100, 200, 300};
    }
    return DeepSeekRankProcessHandle{100U + manifest.rank, 200U + manifest.rank,
                                     300U + manifest.rank};
  }
  Result<DeepSeekRankProcessObservation> observe(
      const DeepSeekRankProcessHandle& handle) override {
    if (failed_process == handle.process_identity)
      return DeepSeekRankProcessObservation::kExitedFailure;
    return DeepSeekRankProcessObservation::kRunning;
  }
  Status terminate(const DeepSeekRankProcessHandle& handle) override {
    terminated.push_back(handle.process_identity); return Status::Ok();
  }
  Status send_challenge(const DeepSeekRankProcessHandle&,
                        const DeepSeekRankExecChallenge& challenge) override {
    if (!challenge_status.ok()) return challenge_status;
    if (challenge.manifest.rank == failed_challenge_rank)
      return Status::Internal("injected rank challenge failure");
    challenges.push_back(challenge); return Status::Ok();
  }
  Result<std::optional<DeepSeekRankExecReady>> poll_ready(
      const DeepSeekRankProcessHandle& handle) override {
    if (ready_pending) return std::optional<DeepSeekRankExecReady>{};
    for (const auto& challenge : challenges) {
      if (challenge.handle.process_identity == handle.process_identity) {
        const auto& m = challenge.manifest;
        return std::optional<DeepSeekRankExecReady>{{
            {m.engine_epoch, m.worker_generation, m.rank,
             m.physical_device_identity, m.process_manifest_identity,
             handle.process_identity, handle.pidfd_identity,
             handle.control_identity, m.physical_device_uuid_commitment,
             m.startup_device_ordinal, m.startup_deadline_ns},
            challenge.challenge_identity}};
      }
    }
    return std::optional<DeepSeekRankExecReady>{};
  }
  std::vector<std::uint32_t> spawned;
  std::vector<std::uint64_t> terminated;
  std::uint64_t failed_process = 0;
  std::vector<DeepSeekRankExecChallenge> challenges;
  Status challenge_status = Status::Ok();
  std::uint32_t failed_challenge_rank =
      std::numeric_limits<std::uint32_t>::max();
  bool ready_pending = false;
  bool duplicate_handles = false;
};

std::vector<DeepSeekRankProcessManifest> manifests(std::uint32_t count) {
  std::vector<DeepSeekRankProcessManifest> result;
  for (std::uint32_t rank = 0; rank < count; ++rank)
    result.push_back({7, 8, count, rank, 10U + rank, 20U + rank,
                      device_commitment(rank),
                      static_cast<std::int32_t>(rank * 2U + 1U), 200});
  return result;
}

DeepSeekRankSpawnResourcePlan resource_plan(
    std::uint32_t count, std::uint64_t node_reserve = 41) {
  return {count, 3, 20, 4, 200, 10, node_reserve};
}

DeepSeekRankSpawnResourceObservation resource_observation() {
  return {10,
          100,
          true,
          {{device_commitment(50), 5, 100},
           {device_commitment(51), 10, 200}},
          10,
          100,
          200,
          1000,
          100,
          1000,
          1000};
}

DeepSeekRankSpawnPreflightReceipt preflight(
    const std::vector<DeepSeekRankProcessManifest>& values,
    std::uint64_t node_reserve = 41) {
  auto compiled = DeepSeekRankSpawnPreflightReceipt::Compile(
      values, resource_plan(static_cast<std::uint32_t>(values.size()),
                            node_reserve),
      resource_observation());
  return std::move(*compiled);
}

DeepSeekRankSpawnAuthorization authorization(
    const std::vector<DeepSeekRankProcessManifest>& values,
    std::uint8_t capacity = 40, std::uint8_t resources = 41) {
  auto receipt = preflight(values, resources);
  auto capacity_instance = test_fixture::rank_capacity_instance(
      values,
      resource_plan(static_cast<std::uint32_t>(values.size()), resources),
      capacity);
  auto created = DeepSeekRankSpawnAuthorization::Create(
      values, std::move(capacity_instance), receipt);
  return std::move(*created);
}

DeepSeekRankProcessSupervisor make_supervisor(
    const std::vector<DeepSeekRankProcessManifest>& values,
    ProcessDriver& driver) {
  auto created = DeepSeekRankProcessSupervisor::Create(
      values, authorization(values), driver);
  return std::move(*created);
}

DeepSeekRankReadyReceipt ready(
    const DeepSeekRankProcessManifest& m) {
  return {m.engine_epoch, m.worker_generation, m.rank,
          m.physical_device_identity, m.process_manifest_identity,
          100U + m.rank, 200U + m.rank, 300U + m.rank,
          m.physical_device_uuid_commitment, m.startup_device_ordinal,
          m.startup_deadline_ns};
}

TEST(DeepSeekRankProcessSupervisorTest, LaunchesAndReconcilesOneToFourRanks) {
  for (std::uint32_t count = 1; count <= 4; ++count) {
    ProcessDriver driver;
    const auto plan = manifests(count);
    auto supervisor = make_supervisor(plan, driver);
    ASSERT_TRUE(supervisor.launch().ok());
    ASSERT_TRUE(supervisor.poll().ok());
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      ASSERT_TRUE(supervisor.dispatch_challenge(rank, 90, 400U + rank).ok());
      ASSERT_TRUE(supervisor.poll_ready(rank).ok());
    }
    EXPECT_TRUE(supervisor.ready());
    EXPECT_EQ(driver.spawned.size(), count);
  }
}

TEST(DeepSeekRankProcessSupervisorTest,
     BatchDispatchAndAdvanceOwnTheCompleteExecStartup) {
  for (std::uint32_t count = 1; count <= 4; ++count) {
    ProcessDriver driver;
    driver.ready_pending = true;
    const auto plan = manifests(count);
    auto supervisor = make_supervisor(plan, driver);
    ASSERT_TRUE(supervisor.launch().ok());
    std::vector<std::uint64_t> challenges;
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      challenges.push_back(500U + rank);
    }
    ASSERT_TRUE(supervisor.dispatch_challenges(90, challenges).ok());
    EXPECT_EQ(supervisor.advance_exec_startup(199).code(),
              StatusCode::kUnavailable);
    driver.ready_pending = false;
    EXPECT_TRUE(supervisor.advance_exec_startup(199).ok());
    EXPECT_TRUE(supervisor.ready());
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      EXPECT_TRUE(supervisor.rank_exec_ready(rank));
    }
  }
}

TEST(DeepSeekRankProcessSupervisorTest,
     BatchDispatchRejectsDuplicateIdentityBeforeAnyChallengeSend) {
  ProcessDriver driver;
  const auto plan = manifests(2);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  const std::array<std::uint64_t, 2> duplicate{500, 500};
  EXPECT_FALSE(supervisor.dispatch_challenges(90, duplicate).ok());
  EXPECT_TRUE(driver.challenges.empty());
  EXPECT_TRUE(supervisor.failed());
  EXPECT_EQ(driver.terminated.size(), 2U);
}

TEST(DeepSeekRankProcessSupervisorTest,
     BatchDispatchFailureTerminatesThePartiallyChallengedGeneration) {
  ProcessDriver driver;
  driver.failed_challenge_rank = 1;
  const auto plan = manifests(3);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  const std::array<std::uint64_t, 3> challenges{500, 501, 502};
  EXPECT_FALSE(supervisor.dispatch_challenges(90, challenges).ok());
  ASSERT_EQ(driver.challenges.size(), 1U);
  EXPECT_EQ(driver.challenges[0].manifest.rank, 0U);
  EXPECT_TRUE(supervisor.failed());
  EXPECT_EQ(driver.terminated.size(), 3U);
}

TEST(DeepSeekRankProcessSupervisorTest,
     AdvanceRejectsMissingChallengeAndDeadlineAtomically) {
  {
    ProcessDriver driver;
    const auto plan = manifests(2);
    auto supervisor = make_supervisor(plan, driver);
    ASSERT_TRUE(supervisor.launch().ok());
    EXPECT_FALSE(supervisor.advance_exec_startup(199).ok());
    EXPECT_TRUE(supervisor.failed());
    EXPECT_EQ(driver.terminated.size(), 2U);
  }
  {
    ProcessDriver driver;
    const auto plan = manifests(2);
    auto supervisor = make_supervisor(plan, driver);
    ASSERT_TRUE(supervisor.launch().ok());
    const std::array<std::uint64_t, 2> challenges{500, 501};
    ASSERT_TRUE(supervisor.dispatch_challenges(90, challenges).ok());
    EXPECT_EQ(supervisor.advance_exec_startup(200).code(),
              StatusCode::kDeadlineExceeded);
    EXPECT_TRUE(supervisor.failed());
    EXPECT_EQ(driver.terminated.size(), 2U);
  }
}

TEST(DeepSeekRankProcessSupervisorTest, FailsWholeGenerationOnWorkerExit) {
  ProcessDriver driver;
  const auto plan = manifests(3);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  driver.failed_process = 101;
  EXPECT_FALSE(supervisor.poll().ok());
  EXPECT_TRUE(supervisor.failed());
  EXPECT_EQ(driver.terminated,
            (std::vector<std::uint64_t>{100, 101, 102}));
}

TEST(DeepSeekRankProcessSupervisorTest,
     RejectsDuplicateSpawnHandlesAndTerminatesWholeGeneration) {
  ProcessDriver driver;
  driver.duplicate_handles = true;
  const auto plan = manifests(2);
  auto supervisor = make_supervisor(plan, driver);

  EXPECT_FALSE(supervisor.launch().ok());
  EXPECT_TRUE(supervisor.failed());
  EXPECT_EQ(driver.spawned, (std::vector<std::uint32_t>{0, 1}));
  EXPECT_EQ(driver.terminated, (std::vector<std::uint64_t>{100, 100}));
}

TEST(DeepSeekRankProcessSupervisorTest, RejectsReadyReplayAndDeviceDrift) {
  ProcessDriver driver;
  const auto plan = manifests(2);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  ASSERT_TRUE(supervisor.dispatch_challenge(0, 90, 400).ok());
  driver.challenges[0].manifest.physical_device_identity++;
  EXPECT_FALSE(supervisor.poll_ready(0).ok());
  EXPECT_TRUE(supervisor.failed());
  EXPECT_EQ(driver.terminated.size(), 2U);
}

TEST(DeepSeekRankProcessSupervisorTest, RejectsUuidCommitmentDrift) {
  ProcessDriver driver;
  const auto plan = manifests(1);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  ASSERT_TRUE(supervisor.dispatch_challenge(0, 90, 400).ok());
  driver.challenges[0].manifest.physical_device_uuid_commitment.bytes[0] =
      std::byte{9};
  EXPECT_FALSE(supervisor.poll_ready(0).ok());
  EXPECT_TRUE(supervisor.failed());
}

TEST(DeepSeekRankProcessSupervisorTest, RejectsUnboundOrReplayedChallenge) {
  ProcessDriver driver;
  const auto plan = manifests(1);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  EXPECT_FALSE(supervisor.poll_ready(0).ok());
  EXPECT_FALSE(supervisor.failed());
}

TEST(DeepSeekRankProcessSupervisorTest, PendingReadyDoesNotPoisonGeneration) {
  ProcessDriver driver;
  driver.ready_pending = true;
  const auto plan = manifests(1);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  ASSERT_TRUE(supervisor.dispatch_challenge(0, 90, 400).ok());
  EXPECT_EQ(supervisor.poll_ready(0).code(), StatusCode::kUnavailable);
  EXPECT_FALSE(supervisor.failed());
  driver.ready_pending = false;
  EXPECT_TRUE(supervisor.poll_ready(0).ok());
  EXPECT_TRUE(supervisor.ready());
}

TEST(DeepSeekRankProcessSupervisorTest, ChallengeFailureTerminatesGeneration) {
  ProcessDriver driver;
  driver.challenge_status = Status::Internal("injected channel failure");
  const auto plan = manifests(2);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  EXPECT_FALSE(supervisor.dispatch_challenge(0, 90, 400).ok());
  EXPECT_TRUE(supervisor.failed());
  EXPECT_EQ(driver.terminated.size(), 2U);
}

TEST(DeepSeekRankProcessSupervisorTest, AbsoluteDeadlineFailsWholeGeneration) {
  ProcessDriver driver;
  const auto plan = manifests(2);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  EXPECT_TRUE(supervisor.expire(199).ok());
  EXPECT_EQ(supervisor.expire(200).code(), StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(supervisor.failed());
  EXPECT_EQ(driver.terminated.size(), 2U);
}

TEST(DeepSeekRankProcessSupervisorTest, CompletedReadyIgnoresStartupExpiry) {
  ProcessDriver driver;
  const auto plan = manifests(1);
  auto supervisor = make_supervisor(plan, driver);
  ASSERT_TRUE(supervisor.launch().ok());
  ASSERT_TRUE(supervisor.dispatch_challenge(0, 90, 400).ok());
  ASSERT_TRUE(supervisor.poll_ready(0).ok());
  EXPECT_TRUE(supervisor.expire(200).ok());
  EXPECT_TRUE(supervisor.ready());
}

TEST(DeepSeekRankProcessSupervisorTest, RejectsDuplicatePhysicalGpu) {
  auto plan = manifests(2);
  plan[1].physical_device_identity = plan[0].physical_device_identity;
  EXPECT_FALSE(DeepSeekRankSpawnPreflightReceipt::Compile(
                   plan, resource_plan(2), resource_observation()).ok());
}

TEST(DeepSeekRankProcessSupervisorTest, RejectsDuplicateStartupOrdinal) {
  auto plan = manifests(2);
  plan[1].startup_device_ordinal = plan[0].startup_device_ordinal;
  EXPECT_FALSE(DeepSeekRankSpawnPreflightReceipt::Compile(
                   plan, resource_plan(2), resource_observation()).ok());
}

TEST(DeepSeekRankProcessSupervisorTest,
     RejectsForeignAuthorizationBeforeAnySpawn) {
  ProcessDriver driver;
  const auto authorized_plan = manifests(2);
  auto changed_plan = authorized_plan;
  changed_plan[1].process_manifest_identity++;
  auto created = DeepSeekRankProcessSupervisor::Create(
      changed_plan, authorization(authorized_plan), driver);
  EXPECT_FALSE(created.ok());
  EXPECT_TRUE(driver.spawned.empty());
}

TEST(DeepSeekRankProcessSupervisorTest,
     AuthorizationBindsCapacityAndResourceRoots) {
  EXPECT_EQ(kDeepSeekRankSpawnAuthorizationAbi,
            "pih_deepseek_rank_spawn_authorization_v1");
  EXPECT_EQ(kDeepSeekRankSpawnPreflightAbi,
            "pih_deepseek_rank_spawn_preflight_v1");
  const auto plan = manifests(2);
  const auto first = authorization(plan, 40, 41);
  const auto capacity_changed = authorization(plan, 42, 41);
  const auto resource_changed = authorization(plan, 40, 43);
  EXPECT_NE(first.authorization_root(), capacity_changed.authorization_root());
  EXPECT_NE(first.authorization_root(), resource_changed.authorization_root());
  EXPECT_EQ(first.authorization_root().hex(),
            "b865ac169c8bd44bd024851577986b3cf1d593ebb4c48c36e8c409cf538912ff");
  EXPECT_EQ(first.engine_epoch(), 7U);
  EXPECT_EQ(first.worker_generation(), 8U);
  EXPECT_EQ(first.world_size(), 2U);
}

TEST(DeepSeekRankProcessSupervisorTest,
     AuthorizationRejectsCapacityAndPreflightPlanSplice) {
  const auto manifests_value = manifests(2);
  auto capacity_instance = test_fixture::rank_capacity_instance(
      manifests_value, resource_plan(2, 41));
  auto foreign_preflight = preflight(manifests_value, 43);

  EXPECT_FALSE(DeepSeekRankSpawnAuthorization::Create(
                   manifests_value, std::move(capacity_instance),
                   foreign_preflight)
                   .ok());
}

TEST(DeepSeekRankProcessSupervisorTest,
     PreflightDerivesImmediateSpawnPeaksAndStableRoot) {
  const auto plan = manifests(2);
  const auto receipt = preflight(plan);
  EXPECT_EQ(receipt.task_increment(), 5U);
  EXPECT_EQ(receipt.controller_fd_increment(), 10U);
  EXPECT_EQ(receipt.node_file_handle_increment(), 49U);
  EXPECT_EQ(receipt.receipt_root().hex(),
            "0ab47700c2cf4714cca51edb8765befab0f82d3eb5ce7fa66e29714f262e7fd7");
}

TEST(DeepSeekRankProcessSupervisorTest,
     PreflightAcceptsExactHeadroomAndRejectsOneShort) {
  const auto manifests_value = manifests(2);
  const auto plan = resource_plan(2);

  auto exact = resource_observation();
  exact.uid_tasks_current = 95;
  exact.cgroup_ancestors[0].current_tasks = 95;
  exact.cgroup_ancestors[1].current_tasks = 95;
  exact.controller_open_fds = 90;
  exact.node_file_allocated = 951;
  exact.vm_max_map_count = 210;
  ASSERT_TRUE(DeepSeekRankSpawnPreflightReceipt::Compile(
                  manifests_value, plan, exact).ok());

  auto one_short = exact;
  one_short.uid_tasks_current = 96;
  EXPECT_EQ(DeepSeekRankSpawnPreflightReceipt::Compile(
                manifests_value, plan, one_short).status().code(),
            StatusCode::kResourceExhausted);
  one_short = exact;
  one_short.cgroup_ancestors[0].current_tasks = 96;
  one_short.cgroup_ancestors[1].current_tasks = 96;
  EXPECT_EQ(DeepSeekRankSpawnPreflightReceipt::Compile(
                manifests_value, plan, one_short).status().code(),
            StatusCode::kResourceExhausted);
  one_short = exact;
  one_short.controller_open_fds = 91;
  EXPECT_EQ(DeepSeekRankSpawnPreflightReceipt::Compile(
                manifests_value, plan, one_short).status().code(),
            StatusCode::kResourceExhausted);
  one_short = exact;
  one_short.node_file_allocated = 952;
  EXPECT_EQ(DeepSeekRankSpawnPreflightReceipt::Compile(
                manifests_value, plan, one_short).status().code(),
            StatusCode::kResourceExhausted);
  one_short = exact;
  one_short.vm_max_map_count = 209;
  EXPECT_EQ(DeepSeekRankSpawnPreflightReceipt::Compile(
                manifests_value, plan, one_short).status().code(),
            StatusCode::kResourceExhausted);
  one_short = resource_observation();
  one_short.rlimit_nofile_soft = 23;
  EXPECT_EQ(DeepSeekRankSpawnPreflightReceipt::Compile(
                manifests_value, plan, one_short).status().code(),
            StatusCode::kResourceExhausted);
}

TEST(DeepSeekRankProcessSupervisorTest,
     PreflightHandlesUnlimitedScopesWithoutRemovingOtherGates) {
  const auto manifests_value = manifests(2);
  const auto plan = resource_plan(2);
  auto observation = resource_observation();
  observation.rlimit_nproc_soft.reset();
  observation.cgroup_ancestors[0].maximum_tasks.reset();
  ASSERT_TRUE(DeepSeekRankSpawnPreflightReceipt::Compile(
                  manifests_value, plan, observation).ok());
  observation.node_file_allocated = 1000;
  EXPECT_EQ(DeepSeekRankSpawnPreflightReceipt::Compile(
                manifests_value, plan, observation).status().code(),
            StatusCode::kResourceExhausted);
}

TEST(DeepSeekRankProcessSupervisorTest,
     PreflightRejectsManifestCountAndCgroupShapeDrift) {
  const auto manifests_value = manifests(2);
  auto plan = resource_plan(1);
  auto observation = resource_observation();
  EXPECT_FALSE(DeepSeekRankSpawnPreflightReceipt::Compile(
                   manifests_value, plan, observation).ok());
  plan = resource_plan(2);
  observation.cgroup_ancestors[1].scope_root =
      observation.cgroup_ancestors[0].scope_root;
  EXPECT_FALSE(DeepSeekRankSpawnPreflightReceipt::Compile(
                   manifests_value, plan, observation).ok());
  observation = resource_observation();
  observation.cgroup_ancestors[1].current_tasks = 4;
  EXPECT_FALSE(DeepSeekRankSpawnPreflightReceipt::Compile(
                   manifests_value, plan, observation).ok());
}

TEST(DeepSeekRankProcessSupervisorTest,
     PreflightRejectsCheckedResourceOverflow) {
  const auto manifests_value = manifests(1);
  auto plan = resource_plan(1);
  plan.worker_fd_peak = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(DeepSeekRankSpawnPreflightReceipt::Compile(
                manifests_value, plan, resource_observation()).status().code(),
            StatusCode::kResourceExhausted);
}

TEST(DeepSeekRankProcessSupervisorTest,
     SupervisorRetainsIdentityButConsumesLaunchAuthorization) {
  ProcessDriver driver;
  const auto plan = manifests(1);
  auto permit = authorization(plan);
  const auto expected_root = permit.authorization_root();
  auto created = DeepSeekRankProcessSupervisor::Create(
      plan, std::move(permit), driver);
  ASSERT_TRUE(created.ok());
  EXPECT_EQ(created->spawn_authorization_root(), expected_root);
  EXPECT_TRUE(created->capacity_authority_retained());
  ASSERT_TRUE(created->launch().ok());
  EXPECT_TRUE(created->capacity_authority_retained());
  EXPECT_FALSE(created->launch().ok());
  EXPECT_EQ(driver.spawned.size(), 1U);
}

TEST(DeepSeekRankProcessSupervisorTest, RejectsMovedFromCapacityInstance) {
  const auto plan = manifests(1);
  auto receipt = preflight(plan);
  auto moved_from_capacity = test_fixture::rank_capacity_instance(
      plan, resource_plan(1));
  auto retained_capacity = std::move(moved_from_capacity);
  EXPECT_FALSE(DeepSeekRankSpawnAuthorization::Create(
                   plan, std::move(moved_from_capacity), receipt)
                   .ok());
  EXPECT_NE(retained_capacity.instance_root(), Sha256Digest{});
}

}  // namespace
}  // namespace pih
