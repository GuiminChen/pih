#include "pih/model/deepseek_rank_startup_barrier.h"

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "deepseek_rank_capacity_test_fixture.h"

namespace pih {
namespace {

Sha256Digest barrier_digest(std::uint8_t seed) {
  Sha256Digest value{};
  value.bytes.fill(static_cast<std::byte>(seed));
  return value;
}

std::vector<DeepSeekRankProcessManifest> barrier_manifests(
    std::uint32_t world_size) {
  std::vector<DeepSeekRankProcessManifest> manifests;
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    manifests.push_back(
        {7, 8, world_size, rank, 10U + rank, 20U + rank,
         barrier_digest(static_cast<std::uint8_t>(rank + 1)),
         static_cast<std::int32_t>(rank), 500});
  }
  return manifests;
}

DeepSeekRankSpawnResourcePlan barrier_spawn_plan(std::uint32_t world_size) {
  return {world_size, 3, 20, 4, 200, 10, 41};
}

DeepSeekRankPostExecResourcePlan barrier_post_exec_plan() {
  return {8, 20, 200, 2, 4, 10, barrier_digest(30)};
}

class BarrierCollector final : public DeepSeekRankSpawnResourceCollector {
 public:
  Result<DeepSeekRankSpawnResourceObservation> collect() override {
    return DeepSeekRankSpawnResourceObservation{
        10,
        100,
        true,
        {{barrier_digest(50), 5, 100},
         {barrier_digest(51), 10, 200}},
        10,
        100,
        200,
        1000,
        100,
        1000,
        1000};
  }
};

class BarrierDriver final : public DeepSeekRankProcessDriver,
                            public DeepSeekRankPostExecResourceChannel {
 public:
  explicit BarrierDriver(std::uint32_t world_size)
      : challenges(world_size), authorities(world_size),
        ready_pending(world_size), observation_pending(world_size) {}

  Result<DeepSeekRankProcessHandle> spawn(
      const DeepSeekRankProcessManifest& manifest) override {
    return DeepSeekRankProcessHandle{100U + manifest.rank,
                                     200U + manifest.rank,
                                     300U + manifest.rank};
  }

  Result<DeepSeekRankProcessObservation> observe(
      const DeepSeekRankProcessHandle& handle) override {
    if (failed_process == handle.process_identity) {
      return DeepSeekRankProcessObservation::kExitedFailure;
    }
    return DeepSeekRankProcessObservation::kRunning;
  }

  Status terminate(const DeepSeekRankProcessHandle& handle) override {
    terminated.push_back(handle.process_identity);
    return Status::Ok();
  }

  Status send_challenge(
      const DeepSeekRankProcessHandle& handle,
      const DeepSeekRankExecChallenge& challenge) override {
    challenges[rank_of(handle)] = challenge;
    return Status::Ok();
  }

  Result<std::optional<DeepSeekRankExecReady>> poll_ready(
      const DeepSeekRankProcessHandle& handle) override {
    const auto rank = rank_of(handle);
    if (ready_pending[rank] > 0) {
      --ready_pending[rank];
      return std::optional<DeepSeekRankExecReady>{};
    }
    if (!challenges[rank]) {
      return Status::FailedPrecondition("test challenge is absent");
    }
    const auto& challenge = *challenges[rank];
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

  Status send_authority(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) override {
    auto decoded = decode_deepseek_rank_post_exec_resource_authority(frame);
    if (!decoded.ok()) return decoded.status();
    authorities[rank_of(handle)] = std::move(*decoded);
    return Status::Ok();
  }

  Result<std::optional<DeepSeekRankPostExecResourceObservation>>
  poll_observation(const DeepSeekRankProcessHandle& handle) override {
    const auto rank = rank_of(handle);
    if (observation_pending[rank] > 0) {
      --observation_pending[rank];
      return std::optional<DeepSeekRankPostExecResourceObservation>{};
    }
    if (!authorities[rank]) {
      return Status::FailedPrecondition("test authority is absent");
    }
    const auto& authority = *authorities[rank];
    auto envelope = authority.os_resource_envelope_root;
    if (rank == foreign_observation_rank) envelope = barrier_digest(99);
    return std::optional<DeepSeekRankPostExecResourceObservation>{{
        authority.engine_epoch,
        authority.worker_generation,
        authority.rank,
        authority.process_identity,
        authority.challenge_identity,
        authority.capacity_plan_instance_root,
        envelope,
        8,
        20,
        0,
        200,
        24,
        100,
        1000,
        210,
        true}};
  }

  static std::size_t rank_of(const DeepSeekRankProcessHandle& handle) {
    return static_cast<std::size_t>(handle.process_identity - 100U);
  }

  std::vector<std::optional<DeepSeekRankExecChallenge>> challenges;
  std::vector<std::optional<DeepSeekRankPostExecResourceAuthority>>
      authorities;
  std::vector<std::uint32_t> ready_pending;
  std::vector<std::uint32_t> observation_pending;
  std::size_t foreign_observation_rank =
      std::numeric_limits<std::size_t>::max();
  std::uint64_t failed_process = 0;
  std::vector<std::uint64_t> terminated;
};

DeepSeekRankSpawnCoordinator create_spawn_coordinator(
    const std::vector<DeepSeekRankProcessManifest>& manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    BarrierCollector& collector, BarrierDriver& driver) {
  auto capacity = test_fixture::rank_capacity_instance(
      manifests, spawn_plan, 90);
  return DeepSeekRankSpawnCoordinator::Create(
             manifests, spawn_plan, std::move(capacity), collector, driver)
      .value();
}

void launch_and_challenge(DeepSeekRankSpawnCoordinator& coordinator,
                          std::uint32_t world_size) {
  ASSERT_TRUE(coordinator.launch().ok());
  std::vector<std::uint64_t> challenges;
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    challenges.push_back(400U + rank);
  }
  ASSERT_TRUE(
      coordinator.supervisor()->dispatch_challenges(90, challenges).ok());
}

TEST(DeepSeekRankStartupBarrierTest, SealsOneToFourRanksBeforeReady) {
  EXPECT_EQ(kDeepSeekRankStartupBarrierAbi,
            "pih_deepseek_rank_startup_barrier_v1");
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    const auto manifests = barrier_manifests(world_size);
    const auto spawn_plan = barrier_spawn_plan(world_size);
    const std::vector<DeepSeekRankPostExecResourcePlan> post_exec_plans(
        world_size, barrier_post_exec_plan());
    BarrierCollector collector;
    BarrierDriver driver(world_size);
    for (auto& pending : driver.ready_pending) pending = 1;
    auto coordinator = create_spawn_coordinator(
        manifests, spawn_plan, collector, driver);
    auto barrier = DeepSeekRankStartupBarrier::Create(
        coordinator, manifests, spawn_plan, post_exec_plans, 700, driver)
                       .value();
    launch_and_challenge(coordinator, world_size);

    EXPECT_EQ(barrier.advance(400).code(), StatusCode::kUnavailable);
    EXPECT_FALSE(barrier.exec_ready());
    EXPECT_FALSE(barrier.ready());
    ASSERT_TRUE(barrier.advance(401).ok());
    EXPECT_TRUE(barrier.exec_ready());
    EXPECT_TRUE(barrier.ready());
    ASSERT_NE(barrier.resource_seal(), nullptr);
    EXPECT_EQ(barrier.resource_seal()->world_size(), world_size);
    EXPECT_TRUE(driver.terminated.empty());
  }
}

TEST(DeepSeekRankStartupBarrierTest,
     ExecReadyIsDiagnosticOnlyWhileResourcesArePending) {
  const auto manifests = barrier_manifests(2);
  const auto spawn_plan = barrier_spawn_plan(2);
  const std::vector<DeepSeekRankPostExecResourcePlan> post_exec_plans(
      2, barrier_post_exec_plan());
  BarrierCollector collector;
  BarrierDriver driver(2);
  driver.observation_pending[1] = 1;
  auto coordinator = create_spawn_coordinator(
      manifests, spawn_plan, collector, driver);
  auto barrier = DeepSeekRankStartupBarrier::Create(
      coordinator, manifests, spawn_plan, post_exec_plans, 700, driver)
                     .value();
  launch_and_challenge(coordinator, 2);

  EXPECT_EQ(barrier.advance(400).code(), StatusCode::kUnavailable);
  EXPECT_TRUE(barrier.exec_ready());
  EXPECT_FALSE(barrier.ready());
  EXPECT_EQ(barrier.resource_seal(), nullptr);
  ASSERT_TRUE(barrier.advance(401).ok());
  EXPECT_TRUE(barrier.ready());
}

TEST(DeepSeekRankStartupBarrierTest,
     ForeignResourceObservationPoisonsAndTerminatesAllRanks) {
  const auto manifests = barrier_manifests(2);
  const auto spawn_plan = barrier_spawn_plan(2);
  const std::vector<DeepSeekRankPostExecResourcePlan> post_exec_plans(
      2, barrier_post_exec_plan());
  BarrierCollector collector;
  BarrierDriver driver(2);
  driver.foreign_observation_rank = 1;
  auto coordinator = create_spawn_coordinator(
      manifests, spawn_plan, collector, driver);
  auto barrier = DeepSeekRankStartupBarrier::Create(
      coordinator, manifests, spawn_plan, post_exec_plans, 700, driver)
                     .value();
  launch_and_challenge(coordinator, 2);

  EXPECT_FALSE(barrier.advance(400).ok());
  EXPECT_TRUE(barrier.poisoned());
  EXPECT_FALSE(barrier.ready());
  EXPECT_EQ(driver.terminated,
            (std::vector<std::uint64_t>{100, 101}));
}

TEST(DeepSeekRankStartupBarrierTest,
     WorkerExitDuringPostExecWaitPoisonsAndTerminatesAllRanks) {
  const auto manifests = barrier_manifests(2);
  const auto spawn_plan = barrier_spawn_plan(2);
  const std::vector<DeepSeekRankPostExecResourcePlan> post_exec_plans(
      2, barrier_post_exec_plan());
  BarrierCollector collector;
  BarrierDriver driver(2);
  // Keep the second worker's resource report pending.  A controller that only
  // waits on the seqpacket would otherwise delay failure until its deadline.
  driver.observation_pending[1] = 1;
  auto coordinator = create_spawn_coordinator(
      manifests, spawn_plan, collector, driver);
  auto barrier = DeepSeekRankStartupBarrier::Create(
      coordinator, manifests, spawn_plan, post_exec_plans, 700, driver)
                     .value();
  launch_and_challenge(coordinator, 2);

  ASSERT_EQ(barrier.advance(400).code(), StatusCode::kUnavailable);
  EXPECT_TRUE(barrier.exec_ready());
  EXPECT_FALSE(barrier.ready());

  driver.failed_process = 101;
  EXPECT_EQ(barrier.advance(401).code(), StatusCode::kUnavailable);
  EXPECT_TRUE(barrier.poisoned());
  EXPECT_FALSE(barrier.ready());
  EXPECT_EQ(driver.terminated,
            (std::vector<std::uint64_t>{100, 101}));
}

TEST(DeepSeekRankStartupBarrierTest,
     RejectsDeadlineBeforeAnyRankResourceWindow) {
  const auto manifests = barrier_manifests(1);
  const auto spawn_plan = barrier_spawn_plan(1);
  const std::vector<DeepSeekRankPostExecResourcePlan> post_exec_plans(
      1, barrier_post_exec_plan());
  BarrierCollector collector;
  BarrierDriver driver(1);
  auto coordinator = create_spawn_coordinator(
      manifests, spawn_plan, collector, driver);

  auto barrier = DeepSeekRankStartupBarrier::Create(
      coordinator, manifests, spawn_plan, post_exec_plans, 500, driver);

  EXPECT_FALSE(barrier.ok());
  EXPECT_EQ(barrier.status().code(), StatusCode::kInvalidArgument);
  EXPECT_FALSE(coordinator.launch_attempted());
}

}  // namespace
}  // namespace pih
