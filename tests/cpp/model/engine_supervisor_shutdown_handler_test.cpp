#include "pih/model/engine_supervisor_shutdown_handler.h"

#include <gtest/gtest.h>

#include <limits>

namespace pih {
namespace {

Sha256Digest lease_digest() {
  Sha256Digest value{};
  value.bytes.fill(std::byte{4});
  return value;
}

EngineAllocationLeaseExpectation expectation() {
  return {lease_digest(), 7, 11};
}

EngineAllocationLeaseObservation observation() {
  return {lease_digest(), 7, 11, true, true};
}

EngineSupervisionCoordinator ready_coordinator() {
  auto value = EngineSupervisionCoordinator::Create(
      3, 1, 100, {10, 100, 1'000, 3}, expectation());
  EXPECT_TRUE(value.ok());
  auto coordinator = std::move(*value);
  EXPECT_TRUE(coordinator.begin_start(observation(), 100).ok());
  EXPECT_TRUE(coordinator.publish_ready(observation()).ok());
  return coordinator;
}

EngineSupervisorShutdownRequest request() {
  return {3, 9, 1, 500, EngineSupervisorShutdownRequestKind::kForceStop};
}

class DomainDriver final : public EngineGenerationDomainDriver {
 public:
  Status force_kill_domain() override {
    ++kills;
    return kill_status;
  }
  Result<bool> domain_empty() override { return false; }

  int kills = 0;
  Status kill_status = Status::Ok();
};

TEST(EngineSupervisorShutdownHandlerTest,
     FirstRequestImmediatelyAuthorizesGenerationForceStop) {
  auto coordinator = ready_coordinator();
  auto shutdown = EngineShutdownController::Create({50, 25}).value();
  auto handler = EngineSupervisorShutdownHandler::Create(3, 9).value();
  DomainDriver driver;
  auto termination = EngineGenerationDomainTermination::Create(driver).value();

  auto result =
      handler.accept(request(), 400, coordinator, shutdown, termination);

  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->ack.disposition,
            EngineSupervisorShutdownAckDisposition::kAccepted);
  EXPECT_EQ(result->ack.acknowledged_ns, 400U);
  EXPECT_EQ(result->action, EngineShutdownAction::kForceKillDomain);
  EXPECT_EQ(coordinator.lifecycle_state(), EngineGenerationState::kStopping);
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kForceStopping);
  EXPECT_EQ(driver.kills, 1);
}

TEST(EngineSupervisorShutdownHandlerTest,
     ExactReplayIsIdempotentAndDoesNotIssueAnotherForceAction) {
  auto coordinator = ready_coordinator();
  auto shutdown = EngineShutdownController::Create({50, 25}).value();
  auto handler = EngineSupervisorShutdownHandler::Create(3, 9).value();
  DomainDriver driver;
  auto termination = EngineGenerationDomainTermination::Create(driver).value();
  auto first =
      handler.accept(request(), 400, coordinator, shutdown, termination);
  ASSERT_TRUE(first.ok()) << first.status().message();

  auto replay =
      handler.accept(request(), 401, coordinator, shutdown, termination);

  ASSERT_TRUE(replay.ok());
  EXPECT_EQ(replay->ack.disposition,
            EngineSupervisorShutdownAckDisposition::kAlreadyStopping);
  EXPECT_EQ(replay->action, EngineShutdownAction::kNone);
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kForceStopping);
  EXPECT_EQ(driver.kills, 1);
}

TEST(EngineSupervisorShutdownHandlerTest,
     RejectsExpiredOrGenerationDriftWithoutMutatingShutdown) {
  auto coordinator = ready_coordinator();
  auto shutdown = EngineShutdownController::Create({50, 25}).value();
  auto handler = EngineSupervisorShutdownHandler::Create(3, 9).value();
  DomainDriver driver;
  auto termination = EngineGenerationDomainTermination::Create(driver).value();
  auto expired = request();
  expired.deadline_ns = 400;

  EXPECT_FALSE(handler.accept(expired, 400, coordinator, shutdown,
                              termination).ok());
  EXPECT_EQ(coordinator.lifecycle_state(), EngineGenerationState::kReady);
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kRunning);
  EXPECT_FALSE(handler.accept(request(), 399, coordinator, shutdown,
                              termination).ok());
}

TEST(EngineSupervisorShutdownHandlerTest,
     FailedForceStopDoesNotConsumeTheOnlyRequest) {
  auto coordinator = ready_coordinator();
  auto shutdown = EngineShutdownController::Create({50, 25}).value();
  auto handler = EngineSupervisorShutdownHandler::Create(3, 9).value();
  DomainDriver driver;
  auto termination = EngineGenerationDomainTermination::Create(driver).value();
  auto bounded = request();
  bounded.deadline_ns = std::numeric_limits<std::uint64_t>::max();

  EXPECT_FALSE(handler.accept(
      bounded, std::numeric_limits<std::uint64_t>::max() - 1,
      coordinator, shutdown, termination).ok());
  EXPECT_EQ(coordinator.lifecycle_state(), EngineGenerationState::kReady);
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kRunning);

  auto retry =
      handler.accept(bounded, 400, coordinator, shutdown, termination);
  ASSERT_TRUE(retry.ok()) << retry.status().message();
  EXPECT_EQ(retry->ack.disposition,
            EngineSupervisorShutdownAckDisposition::kAccepted);
  EXPECT_EQ(retry->action, EngineShutdownAction::kForceKillDomain);
}

TEST(EngineSupervisorShutdownHandlerTest,
     FailedDomainKillCanRetryWithoutReplayingLifecycleTransitions) {
  auto coordinator = ready_coordinator();
  auto shutdown = EngineShutdownController::Create({50, 25}).value();
  auto handler = EngineSupervisorShutdownHandler::Create(3, 9).value();
  DomainDriver driver;
  driver.kill_status = Status::Unavailable("injected domain kill failure");
  auto termination = EngineGenerationDomainTermination::Create(driver).value();

  EXPECT_FALSE(handler.accept(request(), 400, coordinator, shutdown,
                              termination).ok());
  EXPECT_EQ(driver.kills, 1);
  EXPECT_EQ(coordinator.lifecycle_state(), EngineGenerationState::kStopping);
  EXPECT_EQ(shutdown.state(), EngineShutdownState::kForceStopping);

  driver.kill_status = Status::Ok();
  auto retry =
      handler.accept(request(), 401, coordinator, shutdown, termination);
  ASSERT_TRUE(retry.ok()) << retry.status().message();
  EXPECT_EQ(retry->ack.disposition,
            EngineSupervisorShutdownAckDisposition::kAccepted);
  EXPECT_EQ(driver.kills, 2);

  auto replay =
      handler.accept(request(), 402, coordinator, shutdown, termination);
  ASSERT_TRUE(replay.ok());
  EXPECT_EQ(replay->ack.disposition,
            EngineSupervisorShutdownAckDisposition::kAlreadyStopping);
  EXPECT_EQ(driver.kills, 2);
}

}  // namespace
}  // namespace pih
