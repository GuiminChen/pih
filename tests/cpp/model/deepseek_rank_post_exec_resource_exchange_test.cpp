#include "pih/model/deepseek_rank_post_exec_resource_exchange.h"
#include "deepseek_rank_capacity_test_fixture.h"

#include <gtest/gtest.h>

#include <limits>
#include <utility>
#include <vector>

namespace pih {
namespace {

Sha256Digest exchange_digest(std::uint8_t seed) {
  Sha256Digest value{};
  value.bytes.fill(static_cast<std::byte>(seed));
  return value;
}

std::vector<DeepSeekRankProcessManifest> exchange_manifests(
    std::uint32_t count) {
  std::vector<DeepSeekRankProcessManifest> result;
  for (std::uint32_t rank = 0; rank < count; ++rank) {
    result.push_back({7, 8, count, rank, 10U + rank, 20U + rank,
                      exchange_digest(static_cast<std::uint8_t>(rank + 1)),
                      static_cast<std::int32_t>(rank), 500});
  }
  return result;
}

DeepSeekRankSpawnResourcePlan exchange_spawn(std::uint32_t count) {
  return {count, 3, 20, 4, 200, 10, 41};
}

DeepSeekRankSpawnResourceObservation exchange_preflight_observation() {
  return {10,
          100,
          true,
          {{exchange_digest(50), 5, 100},
           {exchange_digest(51), 10, 200}},
          10,
          100,
          200,
          1000,
          100,
          1000,
          1000};
}

DeepSeekRankPostExecResourcePlan exchange_plan() {
  return {8, 20, 200, 2, 4, 10, exchange_digest(30)};
}

class ExchangeDriver final : public DeepSeekRankProcessDriver {
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
  Status terminate(const DeepSeekRankProcessHandle& handle) override {
    terminated.push_back(handle.process_identity);
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
  std::vector<std::uint64_t> terminated;
};

DeepSeekRankProcessSupervisor exchange_supervisor(
    const std::vector<DeepSeekRankProcessManifest>& manifests,
    const DeepSeekRankSpawnResourcePlan& spawn,
    ExchangeDriver& driver) {
  auto preflight = DeepSeekRankSpawnPreflightReceipt::Compile(
      manifests, spawn, exchange_preflight_observation()).value();
  auto capacity = test_fixture::rank_capacity_instance(
      manifests, spawn, 90);
  auto authorization = DeepSeekRankSpawnAuthorization::Create(
      manifests, std::move(capacity), preflight).value();
  auto supervisor = DeepSeekRankProcessSupervisor::Create(
      manifests, std::move(authorization), driver).value();
  (void)supervisor.launch();
  std::vector<std::uint64_t> challenges;
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    challenges.push_back(400U + rank);
  }
  (void)supervisor.dispatch_challenges(90, challenges);
  (void)supervisor.advance_exec_startup(499);
  return supervisor;
}

DeepSeekRankPostExecResourceObservation exchange_observation(
    const DeepSeekRankProcessManifest& manifest,
    const Sha256Digest& capacity_root) {
  return {manifest.engine_epoch, manifest.worker_generation, manifest.rank,
          100U + manifest.rank, 400U + manifest.rank, capacity_root,
          exchange_digest(30), 8, 20, 0, 200, 24, 100, 1000, 210, true};
}

class ExchangeChannel final : public DeepSeekRankPostExecResourceChannel {
 public:
  Status send_authority(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) override {
    const auto rank = static_cast<std::size_t>(handle.process_identity - 100U);
    ++authority_calls[rank];
    if (rank == failed_authority_rank) {
      return Status::Unavailable("injected authority backpressure");
    }
    auto decoded = decode_deepseek_rank_post_exec_resource_authority(frame);
    if (!decoded.ok()) return decoded.status();
    authorities.push_back(std::move(*decoded));
    return Status::Ok();
  }

  Result<std::optional<DeepSeekRankPostExecResourceObservation>>
  poll_observation(const DeepSeekRankProcessHandle& handle) override {
    if (!failure.ok()) return failure;
    const auto rank = static_cast<std::size_t>(handle.process_identity - 100U);
    ++calls[rank];
    if (pending[rank] > 0) {
      --pending[rank];
      return std::optional<DeepSeekRankPostExecResourceObservation>{};
    }
    return std::optional<DeepSeekRankPostExecResourceObservation>{
        observations[rank]};
  }

  std::vector<DeepSeekRankPostExecResourceObservation> observations;
  std::vector<DeepSeekRankPostExecResourceAuthority> authorities;
  std::vector<std::uint32_t> authority_calls;
  std::vector<std::uint32_t> calls;
  std::vector<std::uint32_t> pending;
  std::size_t failed_authority_rank =
      std::numeric_limits<std::size_t>::max();
  Status failure = Status::Ok();
};

ExchangeChannel exchange_channel(
    const std::vector<DeepSeekRankProcessManifest>& manifests,
    const DeepSeekRankProcessSupervisor& supervisor) {
  ExchangeChannel channel;
  for (const auto& manifest : manifests) {
    channel.observations.push_back(exchange_observation(
        manifest, supervisor.capacity_plan_instance_root()));
  }
  channel.calls.resize(manifests.size());
  channel.authority_calls.resize(manifests.size());
  channel.pending.resize(manifests.size());
  return channel;
}

class ReporterProbe final : public DeepSeekRankPostExecResourceProbe {
 public:
  Result<DeepSeekRankPostExecResourceSnapshot> sample() override {
    ++calls;
    return DeepSeekRankPostExecResourceSnapshot{
        100, 8, 20, 0, 200, 24, 100, 1000, 210, true};
  }
  std::uint32_t calls = 0;
};

class ReporterOperations final
    : public DeepSeekRankPostExecResourceReporterOperations {
 public:
  Result<std::optional<std::vector<std::byte>>> receive_authority(
      std::int32_t fd) override {
    ++authority_receives;
    observed_fd = fd;
    if (authority_pending) {
      return std::optional<std::vector<std::byte>>{};
    }
    return std::optional<std::vector<std::byte>>{authority_frame};
  }

  Result<std::uint64_t> monotonic_now_ns() override {
    if (clock_index >= times.size()) {
      return Status::Internal("test clock exhausted");
    }
    return times[clock_index++];
  }
  Status send_observation(
      std::int32_t fd, std::span<const std::byte> frame) override {
    ++sends;
    observed_fd = fd;
    if (pending_sends > 0) {
      --pending_sends;
      return Status::Unavailable("test send pending");
    }
    sent.assign(frame.begin(), frame.end());
    return send_status;
  }

  std::vector<std::uint64_t> times{100, 101, 102};
  std::vector<std::byte> authority_frame;
  std::size_t clock_index = 0;
  std::uint32_t pending_sends = 0;
  std::uint32_t sends = 0;
  std::uint32_t authority_receives = 0;
  bool authority_pending = false;
  std::int32_t observed_fd = -1;
  Status send_status = Status::Ok();
  std::vector<std::byte> sent;
};

DeepSeekRankProcessManifest reporter_manifest() {
  return {7, 8, 1, 0, 10, 11, {}, 5, 50};
}

DeepSeekRankExecReady reporter_ready() {
  return {{7, 8, 0, 10, 11, 100, 200, 300, exchange_digest(7), 5, 50},
          400};
}

DeepSeekRankPostExecResourceAuthority reporter_authority() {
  return {1,
          7,
          8,
          1,
          0,
          11,
          100,
          200,
          300,
          400,
          exchange_digest(40),
          exchange_digest(41),
          exchange_digest(31),
          exchange_digest(30),
          200,
          exchange_plan()};
}

void bind_reporter_authority(ReporterOperations& operations) {
  const auto frame = encode_deepseek_rank_post_exec_resource_authority(
      reporter_authority());
  operations.authority_frame.assign(frame.begin(), frame.end());
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     ReporterRetriesSameFrameWithoutRecollecting) {
  EXPECT_EQ(kDeepSeekRankPostExecResourceExchangeAbi,
            "pih_deepseek_rank_post_exec_resource_exchange_v1");
  ReporterProbe probe;
  auto collector = StableDeepSeekRankPostExecResourceCollector::Create(
      probe).value();
  ReporterOperations operations;
  bind_reporter_authority(operations);
  operations.pending_sends = 1;
  auto reporter = DeepSeekRankPostExecResourceReporter::Create(
      reporter_manifest(), reporter_ready(), 13, collector, operations)
                      .value();

  EXPECT_EQ(reporter.wait_event(),
            DeepSeekRankPostExecResourceReporterWaitEvent::
                kAuthorityReadable);
  EXPECT_FALSE(reporter.deadline_ns().has_value());
  EXPECT_EQ(reporter.advance().code(), StatusCode::kUnavailable);
  EXPECT_FALSE(reporter.poisoned());
  EXPECT_EQ(reporter.wait_event(),
            DeepSeekRankPostExecResourceReporterWaitEvent::
                kObservationWritable);
  EXPECT_EQ(reporter.deadline_ns(), 200U);
  ASSERT_TRUE(reporter.advance().ok());
  EXPECT_TRUE(reporter.reported());
  EXPECT_FALSE(reporter.wait_event().has_value());
  EXPECT_EQ(probe.calls, 2U);
  EXPECT_EQ(operations.sends, 2U);
  EXPECT_EQ(operations.authority_receives, 1U);
  EXPECT_EQ(operations.observed_fd, 13);
  auto decoded = decode_deepseek_rank_post_exec_resource_observation(
      operations.sent);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->challenge_identity, 400U);
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     ReporterRechecksDeadlineAfterCollection) {
  ReporterProbe probe;
  auto collector = StableDeepSeekRankPostExecResourceCollector::Create(
      probe).value();
  ReporterOperations operations;
  bind_reporter_authority(operations);
  operations.times = {100, 200};
  auto reporter = DeepSeekRankPostExecResourceReporter::Create(
      reporter_manifest(), reporter_ready(), 13, collector, operations)
                      .value();

  EXPECT_EQ(reporter.advance().code(), StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(reporter.poisoned());
  EXPECT_EQ(probe.calls, 2U);
  EXPECT_EQ(operations.sends, 0U);
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     ReporterWaitsForAuthorityWithoutClockOrCollection) {
  ReporterProbe probe;
  auto collector = StableDeepSeekRankPostExecResourceCollector::Create(
      probe).value();
  ReporterOperations operations;
  bind_reporter_authority(operations);
  operations.authority_pending = true;
  auto reporter = DeepSeekRankPostExecResourceReporter::Create(
      reporter_manifest(), reporter_ready(), 13, collector, operations)
                      .value();

  EXPECT_EQ(reporter.advance().code(), StatusCode::kUnavailable);
  EXPECT_FALSE(reporter.poisoned());
  EXPECT_EQ(operations.clock_index, 0U);
  EXPECT_EQ(probe.calls, 0U);
  operations.authority_pending = false;
  EXPECT_TRUE(reporter.advance().ok());
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     ReporterRejectsForeignAuthorityBeforeCollection) {
  ReporterProbe probe;
  auto collector = StableDeepSeekRankPostExecResourceCollector::Create(
      probe).value();
  ReporterOperations operations;
  auto authority = reporter_authority();
  authority.challenge_identity++;
  const auto frame = encode_deepseek_rank_post_exec_resource_authority(
      authority);
  operations.authority_frame.assign(frame.begin(), frame.end());
  auto reporter = DeepSeekRankPostExecResourceReporter::Create(
      reporter_manifest(), reporter_ready(), 13, collector, operations)
                      .value();

  EXPECT_FALSE(reporter.advance().ok());
  EXPECT_TRUE(reporter.poisoned());
  EXPECT_EQ(operations.clock_index, 0U);
  EXPECT_EQ(probe.calls, 0U);
  EXPECT_EQ(operations.sends, 0U);
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     CoordinatorSealsOneToFourRankFrames) {
  for (std::uint32_t count = 1; count <= 4; ++count) {
    const auto manifests = exchange_manifests(count);
    const auto spawn = exchange_spawn(count);
    const std::vector<DeepSeekRankPostExecResourcePlan> plans(
        count, exchange_plan());
    ExchangeDriver driver;
    auto supervisor = exchange_supervisor(manifests, spawn, driver);
    auto channel = exchange_channel(manifests, supervisor);
    auto coordinator = DeepSeekRankPostExecResourceCoordinator::Create(
        supervisor, manifests, spawn, plans, 700, channel).value();

    ASSERT_TRUE(coordinator.advance(600).ok());
    EXPECT_TRUE(coordinator.sealed());
    EXPECT_FALSE(coordinator.poisoned());
    EXPECT_EQ(coordinator.receipt_count(), count);
    ASSERT_NE(coordinator.seal(), nullptr);
    EXPECT_EQ(coordinator.seal()->world_size(), count);
    EXPECT_EQ(channel.authorities.size(), count);
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      EXPECT_EQ(channel.authorities[rank].rank, rank);
      EXPECT_EQ(channel.authorities[rank].capacity_plan_instance_root,
                supervisor.capacity_plan_instance_root());
    }
    EXPECT_TRUE(driver.terminated.empty());
  }
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     CoordinatorRetainsPartialReceiptsAcrossPendingPolls) {
  const auto manifests = exchange_manifests(2);
  const auto spawn = exchange_spawn(2);
  const std::vector<DeepSeekRankPostExecResourcePlan> plans(
      2, exchange_plan());
  ExchangeDriver driver;
  auto supervisor = exchange_supervisor(manifests, spawn, driver);
  auto channel = exchange_channel(manifests, supervisor);
  channel.pending[1] = 1;
  auto coordinator = DeepSeekRankPostExecResourceCoordinator::Create(
      supervisor, manifests, spawn, plans, 700, channel).value();

  EXPECT_EQ(coordinator.advance(600).code(), StatusCode::kUnavailable);
  EXPECT_EQ(coordinator.receipt_count(), 1U);
  ASSERT_TRUE(coordinator.advance(601).ok());
  EXPECT_EQ(channel.calls[0], 1U);
  EXPECT_EQ(channel.calls[1], 2U);
  EXPECT_TRUE(coordinator.sealed());
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     CoordinatorForeignObservationAbortsWholeGeneration) {
  const auto manifests = exchange_manifests(2);
  const auto spawn = exchange_spawn(2);
  const std::vector<DeepSeekRankPostExecResourcePlan> plans(
      2, exchange_plan());
  ExchangeDriver driver;
  auto supervisor = exchange_supervisor(manifests, spawn, driver);
  auto channel = exchange_channel(manifests, supervisor);
  channel.observations[1].acknowledged_capacity_plan_instance_root =
      exchange_digest(99);
  auto coordinator = DeepSeekRankPostExecResourceCoordinator::Create(
      supervisor, manifests, spawn, plans, 700, channel).value();

  EXPECT_FALSE(coordinator.advance(600).ok());
  EXPECT_TRUE(coordinator.poisoned());
  EXPECT_TRUE(supervisor.failed());
  EXPECT_EQ(driver.terminated,
            (std::vector<std::uint64_t>{100, 101}));
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     CoordinatorRejectsStaticEnvelopeDriftBeforeChannelPolling) {
  const auto manifests = exchange_manifests(2);
  const auto spawn = exchange_spawn(2);
  std::vector<DeepSeekRankPostExecResourcePlan> plans(2, exchange_plan());
  plans[1].os_resource_envelope_root = exchange_digest(99);
  ExchangeDriver driver;
  auto supervisor = exchange_supervisor(manifests, spawn, driver);
  auto channel = exchange_channel(manifests, supervisor);

  EXPECT_FALSE(DeepSeekRankPostExecResourceCoordinator::Create(
                   supervisor, manifests, spawn, plans, 700, channel)
                   .ok());
  EXPECT_EQ(channel.calls, (std::vector<std::uint32_t>{0, 0}));
  EXPECT_EQ(channel.authority_calls,
            (std::vector<std::uint32_t>{0, 0}));
  EXPECT_TRUE(driver.terminated.empty());
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     CoordinatorAuthorityPartialSendAbortsWithoutObservationPolling) {
  const auto manifests = exchange_manifests(2);
  const auto spawn = exchange_spawn(2);
  const std::vector<DeepSeekRankPostExecResourcePlan> plans(
      2, exchange_plan());
  ExchangeDriver driver;
  auto supervisor = exchange_supervisor(manifests, spawn, driver);
  auto channel = exchange_channel(manifests, supervisor);
  channel.failed_authority_rank = 1;
  auto coordinator = DeepSeekRankPostExecResourceCoordinator::Create(
      supervisor, manifests, spawn, plans, 700, channel).value();

  EXPECT_EQ(coordinator.advance(600).code(), StatusCode::kUnavailable);
  EXPECT_TRUE(coordinator.poisoned());
  EXPECT_TRUE(supervisor.failed());
  EXPECT_EQ(channel.authority_calls,
            (std::vector<std::uint32_t>{1, 1}));
  EXPECT_EQ(channel.calls, (std::vector<std::uint32_t>{0, 0}));
  EXPECT_EQ(driver.terminated,
            (std::vector<std::uint64_t>{100, 101}));
}

TEST(DeepSeekRankPostExecResourceExchangeTest,
     CoordinatorDeadlineAndChannelFailureAbortWholeGeneration) {
  {
    const auto manifests = exchange_manifests(1);
    const auto spawn = exchange_spawn(1);
    const std::vector<DeepSeekRankPostExecResourcePlan> plans(
        1, exchange_plan());
    ExchangeDriver driver;
    auto supervisor = exchange_supervisor(manifests, spawn, driver);
    auto channel = exchange_channel(manifests, supervisor);
    auto coordinator = DeepSeekRankPostExecResourceCoordinator::Create(
        supervisor, manifests, spawn, plans, 700, channel).value();
    EXPECT_EQ(coordinator.advance(700).code(),
              StatusCode::kDeadlineExceeded);
    EXPECT_TRUE(supervisor.failed());
    EXPECT_EQ(channel.calls[0], 0U);
  }
  {
    const auto manifests = exchange_manifests(1);
    const auto spawn = exchange_spawn(1);
    const std::vector<DeepSeekRankPostExecResourcePlan> plans(
        1, exchange_plan());
    ExchangeDriver driver;
    auto supervisor = exchange_supervisor(manifests, spawn, driver);
    auto channel = exchange_channel(manifests, supervisor);
    channel.failure = Status::Internal("injected channel failure");
    auto coordinator = DeepSeekRankPostExecResourceCoordinator::Create(
        supervisor, manifests, spawn, plans, 700, channel).value();
    EXPECT_EQ(coordinator.advance(600).code(), StatusCode::kInternal);
    EXPECT_TRUE(supervisor.failed());
  }
}

}  // namespace
}  // namespace pih
