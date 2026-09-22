#include "pih/model/engine_supervision_coordinator.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest digest() {
  Sha256Digest value{}; value.bytes.fill(std::byte{4}); return value;
}
EngineAllocationLeaseExpectation expectation() { return {digest(), 7, 8}; }
EngineAllocationLeaseObservation lease() { return {digest(), 7, 8, true, true}; }
EngineRestartPolicy restart_policy() { return {10, 40, 100, 3}; }
EngineProgressHeartbeat heartbeat(std::uint64_t progress,
                                  std::uint64_t now) {
  return {9, progress, {progress, progress}, now};
}
EngineGenerationQuiescenceObservation quiescent() {
  return {true, true, true, true, true, true, true, true, true, true};
}
EngineGenerationQuiescenceReceipt receipt(
    EngineGenerationQuiescenceObservation observation,
    std::uint64_t sample, std::uint64_t now) {
  return {9, digest(), sample, now - 1, now, observation};
}
EngineSupervisionCoordinator coordinator() {
  return EngineSupervisionCoordinator::Create(
      9, 2, 10, restart_policy(), expectation()).value();
}

std::vector<EngineSupervisedProcessIdentity> pidfd_identities() {
  return {
      {EngineSupervisedProcessRole::kController, -1, 100, 200},
      {EngineSupervisedProcessRole::kRank, 0, 101, 201},
      {EngineSupervisedProcessRole::kRank, 1, 102, 202}};
}
EnginePidfdReapingLedger completed_pidfds() {
  auto identities = pidfd_identities();
  auto ledger = EnginePidfdReapingLedger::Create(9, identities).value();
  for (std::size_t index = 0; index < identities.size(); ++index)
    ledger.accept({9, index + 1, identities[index], true, true});
  return ledger;
}
EngineGpuProcessInventoryGate empty_gpu_inventory() {
  const std::uint64_t devices[]{10, 20};
  auto gate = EngineGpuProcessInventoryGate::Create(
      9, digest(), devices).value();
  (void)gate.accept(
      {9, digest(), 1, 100, 101, true, {{10, {}}, {20, {}}}}).value();
  return gate;
}
Sha256Digest owned_digest(std::uint8_t value) {
  Sha256Digest result{}; result.bytes.fill(static_cast<std::byte>(value));
  return result;
}
EngineOwnedResourceBaselineGate baseline_resources() {
  std::vector<EngineOwnedResourceBaselineEntry> baselines;
  std::vector<EngineOwnedResourceObservation> observations;
  for (std::uint8_t kind = 0; kind < 6; ++kind) {
    const auto typed = static_cast<EngineOwnedResourceKind>(kind);
    const auto value = owned_digest(kind + 1);
    baselines.push_back({typed, value});
    observations.push_back({typed, true, value});
  }
  auto gate = EngineOwnedResourceBaselineGate::Create(
      9, digest(), baselines).value();
  (void)gate.accept({9, digest(), 1, 100, 101,
                     std::move(observations)}).value();
  return gate;
}

class DomainDriver final : public EngineGenerationDomainDriver {
 public:
  Status force_kill_domain() override { ++kills; return Status::Ok(); }
  Result<bool> domain_empty() override { ++polls; return empty; }
  int kills = 0;
  int polls = 0;
  bool empty = false;
};

TEST(EngineSupervisionCoordinatorTest, PublishesReadyAfterLeaseAndWatchdogArm) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  EXPECT_FALSE(value.ready());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  EXPECT_TRUE(value.ready());
  ASSERT_TRUE(value.accept_heartbeat(heartbeat(1, 101), 101).ok());
  EXPECT_TRUE(value.poll(lease(), true, 105).ok());
  EXPECT_TRUE(value.ready());
}

TEST(EngineSupervisionCoordinatorTest, LeaseDriftWithdrawsReadyAndIsFirst) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  auto drift = lease(); drift.exclusive_ofd_lock_held = false;
  EXPECT_FALSE(value.poll(drift, false, 101).ok());
  EXPECT_FALSE(value.ready());
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kFailed);
  EXPECT_EQ(value.first_failure(),
            EngineSupervisionFailure::kAllocationLeaseDrift);
  EXPECT_FALSE(value.report_rank_loss().ok());
  EXPECT_EQ(value.first_failure(),
            EngineSupervisionFailure::kAllocationLeaseDrift);
}

TEST(EngineSupervisionCoordinatorTest, RankLossBeatsWatchdogInSamePoll) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  EXPECT_FALSE(value.poll(lease(), false, 110).ok());
  EXPECT_EQ(value.first_failure(), EngineSupervisionFailure::kRankLoss);
  EXPECT_FALSE(value.ready());
}

TEST(EngineSupervisionCoordinatorTest, WatchdogTimeoutFailsWholeGeneration) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  EXPECT_EQ(value.poll(lease(), true, 110).code(),
            StatusCode::kDeadlineExceeded);
  EXPECT_EQ(value.first_failure(),
            EngineSupervisionFailure::kProgressWatchdogExpired);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kFailed);
}

TEST(EngineSupervisionCoordinatorTest, RejectsInvalidConfigurationAndBadStart) {
  EXPECT_FALSE(EngineSupervisionCoordinator::Create(
      9, 2, 10, restart_policy(), {}).ok());
  auto value = coordinator();
  auto drift = lease(); ++drift.file_identity;
  EXPECT_FALSE(value.begin_start(drift, 100).ok());
  EXPECT_EQ(value.first_failure(),
            EngineSupervisionFailure::kAllocationLeaseDrift);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kFailed);
}

TEST(EngineSupervisionCoordinatorTest, InvalidEarlyFaultDoesNotCommitFirstError) {
  auto value = coordinator();
  EXPECT_FALSE(value.report_rank_loss().ok());
  EXPECT_EQ(value.first_failure(), EngineSupervisionFailure::kNone);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kLeased);
  EXPECT_FALSE(value.accept_heartbeat(heartbeat(1, 100), 100).ok());
  EXPECT_EQ(value.first_failure(), EngineSupervisionFailure::kNone);
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
}

TEST(EngineSupervisionCoordinatorTest, RetryableFailureWaitsForQuiescence) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_FALSE(value.report_rank_loss().ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto pidfds = completed_pidfds();
  const std::uint64_t devices[]{10, 20};
  auto inventory = EngineGpuProcessInventoryGate::Create(
      9, digest(), devices).value();
  ASSERT_EQ(*inventory.accept(
                {9, digest(), 1, 100, 101, true,
                 {{10, {}}, {20, {999}}}}),
            EngineGpuProcessInventoryState::kOccupied);
  auto resources = baseline_resources();
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 101), pidfds, inventory, resources, 101),
            EngineQuiescenceState::kPending);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kQuiescenceCheck);
  ASSERT_EQ(*inventory.accept(
                {9, digest(), 2, 101, 102, true,
                 {{10, {}}, {20, {}}}}),
            EngineGpuProcessInventoryState::kEmpty);
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 2, 102), pidfds, inventory, resources, 102),
            EngineQuiescenceState::kVerified);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kBackoff);
  EXPECT_EQ(value.retry_not_before_ns(), 112U);
}

TEST(EngineSupervisionCoordinatorTest,
     RestartsOnlyAfterVerifiedQuiescenceBackoffAndFreshWatchdog) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_FALSE(value.report_rank_loss().ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto pidfds = completed_pidfds();
  auto inventory = empty_gpu_inventory();
  auto resources = baseline_resources();
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 102), pidfds, inventory, resources, 102),
            EngineQuiescenceState::kVerified);
  ASSERT_EQ(value.lifecycle_state(), EngineGenerationState::kBackoff);
  EXPECT_FALSE(value.authorize_restart(111, 10, lease()).ok());
  EXPECT_FALSE(value.authorize_restart(112, 11, lease()).ok());
  ASSERT_TRUE(value.authorize_restart(112, 10, lease()).ok());
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kLeased);
  EXPECT_EQ(value.generation(), 10U);
  EXPECT_EQ(value.first_failure(), EngineSupervisionFailure::kNone);
  ASSERT_TRUE(value.begin_start(lease(), 113).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_TRUE(value.accept_heartbeat({10, 1, {1, 1}, 114}, 114).ok());
}

TEST(EngineSupervisionCoordinatorTest,
     RestartLeaseDriftPermanentlyQuarantinesGeneration) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_FALSE(value.report_rank_loss().ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto pidfds = completed_pidfds();
  auto inventory = empty_gpu_inventory();
  auto resources = baseline_resources();
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 102), pidfds, inventory, resources, 102),
            EngineQuiescenceState::kVerified);
  auto drift = lease();
  drift.exclusive_ofd_lock_held = false;
  EXPECT_FALSE(value.authorize_restart(112, 10, drift).ok());
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kQuarantined);
  EXPECT_FALSE(value.authorize_restart(112, 10, lease()).ok());
}

TEST(EngineSupervisionCoordinatorTest, LeaseDriftNeverBecomesAutoRetry) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  auto drift = lease(); drift.descriptor_open = false;
  ASSERT_FALSE(value.publish_ready(drift).ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto pidfds = completed_pidfds();
  auto inventory = empty_gpu_inventory();
  auto resources = baseline_resources();
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 101), pidfds, inventory, resources, 101),
            EngineQuiescenceState::kVerified);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kDead);
  EXPECT_EQ(value.retry_not_before_ns(), 0U);
}

TEST(EngineSupervisionCoordinatorTest, PlannedDrainWithdrawsReadyAndNeverRetries) {
  auto value = coordinator();
  auto shutdown = EngineShutdownController::Create({10, 5}).value();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_EQ(*value.request_termination(shutdown, 101),
            EngineShutdownAction::kWithdrawReadiness);
  EXPECT_FALSE(value.ready());
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kDraining);
  ASSERT_EQ(*value.poll_shutdown(shutdown, 102, false),
            EngineShutdownAction::kBeginStop);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kStopping);
  ASSERT_TRUE(value.mark_shutdown_domain_empty(shutdown).ok());
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kDead);
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto pidfds = completed_pidfds();
  auto inventory = empty_gpu_inventory();
  auto resources = baseline_resources();
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 103), pidfds, inventory, resources, 103),
            EngineQuiescenceState::kVerified);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kDead);
  EXPECT_EQ(value.retry_not_before_ns(), 0U);
  EXPECT_EQ(value.first_failure(), EngineSupervisionFailure::kNone);
}

TEST(EngineSupervisionCoordinatorTest, RepeatedTerminationShortensToForceStop) {
  auto value = coordinator();
  auto shutdown = EngineShutdownController::Create({10, 5}).value();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_TRUE(value.request_termination(shutdown, 101).ok());
  ASSERT_EQ(*value.request_termination(shutdown, 102),
            EngineShutdownAction::kForceKillDomain);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kStopping);
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kForceStopping);
  EXPECT_EQ(*value.request_termination(shutdown, 103),
            EngineShutdownAction::kNone);
}

TEST(EngineSupervisionCoordinatorTest, EarlyTerminationDoesNotSplitStateMachines) {
  auto value = coordinator();
  auto shutdown = EngineShutdownController::Create({10, 5}).value();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  EXPECT_FALSE(value.request_termination(shutdown, 101).ok());
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kStarting);
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kRunning);
}

TEST(EngineSupervisionCoordinatorTest, ExecutesForceKillAndWaitsForEmptyDomain) {
  auto value = coordinator();
  auto shutdown = EngineShutdownController::Create({10, 5}).value();
  DomainDriver driver;
  auto termination = EngineGenerationDomainTermination::Create(driver).value();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_TRUE(value.request_termination(shutdown, 101).ok());
  auto action = value.request_termination(shutdown, 102);
  ASSERT_TRUE(action.ok());
  ASSERT_TRUE(value.execute_force_kill(*action, shutdown, termination).ok());
  EXPECT_EQ(driver.kills, 1);
  auto empty = value.reconcile_shutdown_domain(shutdown, termination);
  ASSERT_TRUE(empty.ok());
  EXPECT_FALSE(*empty);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kStopping);
  driver.empty = true;
  empty = value.reconcile_shutdown_domain(shutdown, termination);
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(*empty);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kDead);
}

TEST(EngineSupervisionCoordinatorTest, RejectsUnissuedForceKillAction) {
  auto value = coordinator();
  auto shutdown = EngineShutdownController::Create({10, 5}).value();
  DomainDriver driver;
  auto termination = EngineGenerationDomainTermination::Create(driver).value();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  EXPECT_FALSE(value.execute_force_kill(
      EngineShutdownAction::kForceKillDomain, shutdown, termination).ok());
  EXPECT_FALSE(value.execute_force_kill(
      EngineShutdownAction::kBeginStop, shutdown, termination).ok());
  EXPECT_EQ(driver.kills, 0);
}

TEST(EngineSupervisionCoordinatorTest, SignalEventsCannotReplayIntoForceStop) {
  auto value = coordinator();
  auto shutdown = EngineShutdownController::Create({10, 5}).value();
  auto gate = EngineTerminationEventGate::Create(9).value();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_EQ(*value.consume_termination_event(
                gate, {9, 1, EngineTerminationEventKind::kSigterm},
                shutdown, 101),
            EngineShutdownAction::kWithdrawReadiness);
  EXPECT_FALSE(value.consume_termination_event(
      gate, {9, 1, EngineTerminationEventKind::kSigterm},
      shutdown, 102).ok());
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kDraining);
}

TEST(EngineSupervisionCoordinatorTest, SecondSignalForcesStopping) {
  auto value = coordinator();
  auto shutdown = EngineShutdownController::Create({10, 5}).value();
  auto gate = EngineTerminationEventGate::Create(9).value();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_TRUE(value.consume_termination_event(
      gate, {9, 1, EngineTerminationEventKind::kSigterm},
      shutdown, 101).ok());
  ASSERT_EQ(*value.consume_termination_event(
                gate, {9, 2, EngineTerminationEventKind::kSigint},
                shutdown, 102),
            EngineShutdownAction::kForceKillDomain);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kStopping);
}

TEST(EngineSupervisionCoordinatorTest, SignalDuringStartupAbortsWithoutDrain) {
  auto value = coordinator();
  auto shutdown = EngineShutdownController::Create({10, 5}).value();
  auto gate = EngineTerminationEventGate::Create(9).value();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_EQ(*value.consume_termination_event(
                gate, {9, 1, EngineTerminationEventKind::kSigterm},
                shutdown, 101),
            EngineShutdownAction::kBeginStop);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kStopping);
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kStopping);
  ASSERT_EQ(*value.consume_termination_event(
                gate, {9, 2, EngineTerminationEventKind::kSigterm},
                shutdown, 102),
            EngineShutdownAction::kForceKillDomain);
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kForceStopping);
}

TEST(EngineSupervisionCoordinatorTest, IncompletePidfdsOverrideClaimedQuiescence) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_FALSE(value.report_rank_loss().ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto identities = pidfd_identities();
  auto pidfds = EnginePidfdReapingLedger::Create(9, identities).value();
  auto inventory = empty_gpu_inventory();
  auto resources = baseline_resources();
  ASSERT_TRUE(pidfds.accept({9, 1, identities[0], true, true}).ok());
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 101), pidfds, inventory, resources, 101),
            EngineQuiescenceState::kPending);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kQuiescenceCheck);
}

TEST(EngineSupervisionCoordinatorTest, PoisonedPidfdsQuarantineGeneration) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_FALSE(value.report_rank_loss().ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto identities = pidfd_identities();
  auto pidfds = EnginePidfdReapingLedger::Create(9, identities).value();
  auto inventory = empty_gpu_inventory();
  auto resources = baseline_resources();
  auto drift = identities[0]; ++drift.pidfd_identity;
  EXPECT_FALSE(pidfds.accept({9, 1, drift, true, true}).ok());
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 101), pidfds, inventory, resources, 101),
            EngineQuiescenceState::kUnknown);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kQuarantined);
}

TEST(EngineSupervisionCoordinatorTest, OccupiedGpuInventoryBlocksQuiescence) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_FALSE(value.report_rank_loss().ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto pidfds = completed_pidfds();
  auto resources = baseline_resources();
  const std::uint64_t devices[]{10, 20};
  auto inventory = EngineGpuProcessInventoryGate::Create(
      9, digest(), devices).value();
  ASSERT_EQ(*inventory.accept(
                {9, digest(), 1, 100, 101, true,
                 {{10, {}}, {20, {999}}}}),
            EngineGpuProcessInventoryState::kOccupied);
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 101), pidfds, inventory, resources, 101),
            EngineQuiescenceState::kPending);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kQuiescenceCheck);
}

TEST(EngineSupervisionCoordinatorTest, UnknownGpuInventoryQuarantinesGeneration) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_FALSE(value.report_rank_loss().ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto pidfds = completed_pidfds();
  auto resources = baseline_resources();
  const std::uint64_t devices[]{10, 20};
  auto inventory = EngineGpuProcessInventoryGate::Create(
      9, digest(), devices).value();
  ASSERT_EQ(*inventory.accept(
                {9, digest(), 1, 100, 101, false,
                 {{10, {}}, {20, {}}}}),
            EngineGpuProcessInventoryState::kUnknown);
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 101), pidfds, inventory, resources, 101),
            EngineQuiescenceState::kUnknown);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kQuarantined);
}

TEST(EngineSupervisionCoordinatorTest, SingleOwnedResourceDriftBlocksQuiescence) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_FALSE(value.report_rank_loss().ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto pidfds = completed_pidfds();
  auto inventory = empty_gpu_inventory();
  std::vector<EngineOwnedResourceBaselineEntry> baselines;
  std::vector<EngineOwnedResourceObservation> observations;
  for (std::uint8_t kind = 0; kind < 6; ++kind) {
    const auto typed = static_cast<EngineOwnedResourceKind>(kind);
    baselines.push_back({typed, owned_digest(kind + 1)});
    observations.push_back({typed, true, owned_digest(kind + 1)});
  }
  observations[static_cast<std::size_t>(
      EngineOwnedResourceKind::kPinnedMemory)].observed_digest = owned_digest(99);
  auto resources = EngineOwnedResourceBaselineGate::Create(
      9, digest(), baselines).value();
  ASSERT_EQ(*resources.accept(
                {9, digest(), 1, 100, 101, observations}),
            EngineOwnedResourceBaselineState::kDrifted);
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 101), pidfds, inventory, resources, 101),
            EngineQuiescenceState::kPending);
  EXPECT_FALSE(resources.at_baseline(
      EngineOwnedResourceKind::kPinnedMemory));
}

TEST(EngineSupervisionCoordinatorTest, UnknownOwnedResourceQuarantinesGeneration) {
  auto value = coordinator();
  ASSERT_TRUE(value.begin_start(lease(), 100).ok());
  ASSERT_TRUE(value.publish_ready(lease()).ok());
  ASSERT_FALSE(value.report_rank_loss().ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
  auto pidfds = completed_pidfds();
  auto inventory = empty_gpu_inventory();
  std::vector<EngineOwnedResourceBaselineEntry> baselines;
  std::vector<EngineOwnedResourceObservation> observations;
  for (std::uint8_t kind = 0; kind < 6; ++kind) {
    const auto typed = static_cast<EngineOwnedResourceKind>(kind);
    baselines.push_back({typed, owned_digest(kind + 1)});
    observations.push_back({typed, true, owned_digest(kind + 1)});
  }
  observations[static_cast<std::size_t>(
      EngineOwnedResourceKind::kNetwork)].visibility_complete = false;
  observations[static_cast<std::size_t>(
      EngineOwnedResourceKind::kNetwork)].observed_digest = {};
  auto resources = EngineOwnedResourceBaselineGate::Create(
      9, digest(), baselines).value();
  ASSERT_EQ(*resources.accept(
                {9, digest(), 1, 100, 101, observations}),
            EngineOwnedResourceBaselineState::kUnknown);
  ASSERT_EQ(*value.observe_quiescence(
                receipt(quiescent(), 1, 101), pidfds, inventory, resources, 101),
            EngineQuiescenceState::kUnknown);
  EXPECT_EQ(value.lifecycle_state(), EngineGenerationState::kQuarantined);
}

}  // namespace
}  // namespace pih
