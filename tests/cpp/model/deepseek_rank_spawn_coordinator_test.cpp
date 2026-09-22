#include "pih/model/deepseek_rank_spawn_coordinator.h"
#include "deepseek_rank_capacity_test_fixture.h"

#include <gtest/gtest.h>

#include <utility>

namespace pih {
namespace {

Sha256Digest digest(std::uint8_t value) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(value));
  return result;
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

DeepSeekRankSpawnResourcePlan plan(std::uint32_t count) {
  return {count, 3, 20, 4, 200, 10, 41};
}

DeepSeekRankSpawnResourceObservation observation() {
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

class Collector final : public DeepSeekRankSpawnResourceCollector {
 public:
  Result<DeepSeekRankSpawnResourceObservation> collect() override {
    ++calls;
    if (!status.ok()) return status;
    return value;
  }

  DeepSeekRankSpawnResourceObservation value = observation();
  Status status = Status::Ok();
  std::uint32_t calls = 0;
};

class Driver final : public DeepSeekRankProcessDriver {
 public:
  Result<DeepSeekRankProcessHandle> spawn(
      const DeepSeekRankProcessManifest& manifest) override {
    spawned.push_back(manifest.rank);
    if (failed_rank && manifest.rank == *failed_rank) {
      return Status::Internal("spawn rejected");
    }
    return DeepSeekRankProcessHandle{100U + manifest.rank,
                                     200U + manifest.rank,
                                     300U + manifest.rank};
  }
  Result<DeepSeekRankProcessObservation> observe(
      const DeepSeekRankProcessHandle&) override {
    return DeepSeekRankProcessObservation::kRunning;
  }
  Status terminate(const DeepSeekRankProcessHandle& handle) override {
    terminated.push_back(handle.process_identity);
    return Status::Ok();
  }
  Status send_challenge(const DeepSeekRankProcessHandle&,
                        const DeepSeekRankExecChallenge&) override {
    return Status::Ok();
  }
  Result<std::optional<DeepSeekRankExecReady>> poll_ready(
      const DeepSeekRankProcessHandle&) override {
    return std::optional<DeepSeekRankExecReady>{};
  }

  std::optional<std::uint32_t> failed_rank;
  std::vector<std::uint32_t> spawned;
  std::vector<std::uint64_t> terminated;
};

TEST(DeepSeekRankSpawnCoordinatorTest,
     CollectsAuthorizesAndLaunchesOneToFourRanksExactlyOnce) {
  for (std::uint32_t count = 1; count <= 4; ++count) {
    Collector collector;
    Driver driver;
    const auto manifest_set = manifests(count);
    auto capacity = test_fixture::rank_capacity_instance(
        manifest_set, plan(count));
    auto coordinator = DeepSeekRankSpawnCoordinator::Create(
        manifest_set, plan(count), std::move(capacity), collector, driver);
    ASSERT_TRUE(coordinator.ok());
    EXPECT_FALSE(coordinator->launch_attempted());
    EXPECT_EQ(coordinator->supervisor(), nullptr);
    ASSERT_TRUE(coordinator->launch().ok());
    EXPECT_TRUE(coordinator->launch_attempted());
    ASSERT_NE(coordinator->supervisor(), nullptr);
    EXPECT_EQ(collector.calls, 1U);
    EXPECT_EQ(driver.spawned.size(), count);
    EXPECT_NE(coordinator->preflight_receipt_root(), Sha256Digest{});
    EXPECT_EQ(coordinator->spawn_authorization_root(),
              coordinator->supervisor()->spawn_authorization_root());
    EXPECT_FALSE(coordinator->launch().ok());
    EXPECT_EQ(collector.calls, 1U);
  }
}

TEST(DeepSeekRankSpawnCoordinatorTest,
     CollectionOrPreflightFailureCannotReachOrRetryDriver) {
  const auto manifest_set = manifests(2);
  Collector unavailable;
  unavailable.status = Status::Unavailable("observation unavailable");
  Driver driver;
  auto unavailable_capacity = test_fixture::rank_capacity_instance(
      manifest_set, plan(2));
  auto collection_failure = DeepSeekRankSpawnCoordinator::Create(
      manifest_set, plan(2), std::move(unavailable_capacity), unavailable,
      driver);
  ASSERT_TRUE(collection_failure.ok());
  EXPECT_EQ(collection_failure->launch().code(), StatusCode::kUnavailable);
  EXPECT_TRUE(driver.spawned.empty());
  EXPECT_FALSE(collection_failure->launch().ok());
  EXPECT_EQ(unavailable.calls, 1U);

  Collector insufficient;
  insufficient.value.rlimit_nofile_soft = 23;
  auto insufficient_capacity = test_fixture::rank_capacity_instance(
      manifest_set, plan(2));
  auto preflight_failure = DeepSeekRankSpawnCoordinator::Create(
      manifest_set, plan(2), std::move(insufficient_capacity), insufficient,
      driver);
  ASSERT_TRUE(preflight_failure.ok());
  EXPECT_EQ(preflight_failure->launch().code(),
            StatusCode::kResourceExhausted);
  EXPECT_TRUE(driver.spawned.empty());
  EXPECT_EQ(insufficient.calls, 1U);
}

TEST(DeepSeekRankSpawnCoordinatorTest,
     DriverFailureConsumesTransactionAndTerminatesPartialGeneration) {
  const auto manifest_set = manifests(2);
  Collector collector;
  Driver driver;
  driver.failed_rank = 1;
  auto capacity = test_fixture::rank_capacity_instance(
      manifest_set, plan(2));
  auto coordinator = DeepSeekRankSpawnCoordinator::Create(
      manifest_set, plan(2), std::move(capacity), collector, driver);
  ASSERT_TRUE(coordinator.ok());
  EXPECT_EQ(coordinator->launch().code(), StatusCode::kInternal);
  EXPECT_EQ(collector.calls, 1U);
  EXPECT_EQ(driver.spawned.size(), 2U);
  EXPECT_EQ(driver.terminated, (std::vector<std::uint64_t>{100}));
  EXPECT_EQ(coordinator->supervisor(), nullptr);
  EXPECT_FALSE(coordinator->launch().ok());
}

TEST(DeepSeekRankSpawnCoordinatorTest,
     RejectsInvalidStaticInputsBeforeCollection) {
  Collector collector;
  Driver driver;
  const auto manifest_set = manifests(2);
  auto wrong_plan_capacity = test_fixture::rank_capacity_instance(
      manifest_set, plan(2));
  EXPECT_FALSE(DeepSeekRankSpawnCoordinator::Create(
                   manifest_set, plan(1), std::move(wrong_plan_capacity),
                   collector, driver)
                   .ok());
  auto invalid_manifest = manifest_set;
  invalid_manifest[1].physical_device_identity =
      invalid_manifest[0].physical_device_identity;
  auto invalid_manifest_capacity = test_fixture::rank_capacity_instance(
      manifest_set, plan(2));
  EXPECT_FALSE(DeepSeekRankSpawnCoordinator::Create(
                   invalid_manifest, plan(2),
                   std::move(invalid_manifest_capacity), collector, driver)
                   .ok());
  auto zero_reserve = plan(2);
  zero_reserve.emergency_task_reserve = 0;
  auto zero_reserve_capacity = test_fixture::rank_capacity_instance(
      manifest_set, plan(2));
  EXPECT_FALSE(DeepSeekRankSpawnCoordinator::Create(
                   manifest_set, zero_reserve,
                   std::move(zero_reserve_capacity), collector, driver)
                   .ok());
  auto moved_from_capacity = test_fixture::rank_capacity_instance(
      manifest_set, plan(2));
  auto retained_capacity = std::move(moved_from_capacity);
  EXPECT_FALSE(DeepSeekRankSpawnCoordinator::Create(
                   manifest_set, plan(2), std::move(moved_from_capacity),
                   collector, driver)
                   .ok());
  EXPECT_NE(retained_capacity.instance_root(), Sha256Digest{});
  EXPECT_EQ(collector.calls, 0U);
}

}  // namespace
}  // namespace pih
