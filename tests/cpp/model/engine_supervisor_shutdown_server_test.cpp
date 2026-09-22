#include "pih/model/engine_supervisor_shutdown_server.h"

#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace pih {
namespace {

Sha256Digest digest() {
  Sha256Digest value{};
  value.bytes.fill(std::byte{6});
  return value;
}

EngineSupervisionCoordinator ready_coordinator() {
  const EngineAllocationLeaseExpectation expectation{digest(), 7, 8};
  const EngineAllocationLeaseObservation observation{
      digest(), 7, 8, true, true};
  auto coordinator = EngineSupervisionCoordinator::Create(
      3, 1, 100, {10, 100, 1'000, 3}, expectation).value();
  EXPECT_TRUE(coordinator.begin_start(observation, 100).ok());
  EXPECT_TRUE(coordinator.publish_ready(observation).ok());
  return coordinator;
}

class ServerDriver final : public EngineSupervisorShutdownServerDriver {
 public:
  Result<std::optional<std::vector<std::byte>>> poll_request() override {
    ++polls;
    auto result = std::move(request);
    request.reset();
    return result;
  }
  Status send_ack(std::span<const std::byte> frame) override {
    ++sends;
    if (!send_status.ok()) return send_status;
    ack.assign(frame.begin(), frame.end());
    return Status::Ok();
  }

  std::optional<std::vector<std::byte>> request;
  std::vector<std::byte> ack;
  Status send_status = Status::Ok();
  int polls = 0;
  int sends = 0;
};

class DomainDriver final : public EngineGenerationDomainDriver {
 public:
  Status force_kill_domain() override {
    ++kills;
    return kill_status;
  }
  Result<bool> domain_empty() override { return false; }

  Status kill_status = Status::Ok();
  int kills = 0;
};

std::vector<std::byte> request_frame() {
  const auto frame = encode_engine_supervisor_shutdown_request(
      {3, 9, 1, 500, EngineSupervisorShutdownRequestKind::kForceStop});
  return {frame.begin(), frame.end()};
}

TEST(EngineSupervisorShutdownServerTest,
     AckBackpressureDoesNotReplayAcceptedDomainKill) {
  ServerDriver driver;
  driver.request = request_frame();
  driver.send_status = Status::Unavailable("injected ACK backpressure");
  auto server = EngineSupervisorShutdownServer::Create(driver).value();
  auto handler = EngineSupervisorShutdownHandler::Create(3, 9).value();
  auto coordinator = ready_coordinator();
  auto shutdown = EngineShutdownController::Create({50, 25}).value();
  DomainDriver domain_driver;
  auto termination =
      EngineGenerationDomainTermination::Create(domain_driver).value();

  auto pending = server.advance(
      400, handler, coordinator, shutdown, termination);
  ASSERT_TRUE(pending.ok());
  EXPECT_FALSE(*pending);
  EXPECT_EQ(domain_driver.kills, 1);
  EXPECT_EQ(driver.polls, 1);
  EXPECT_EQ(driver.sends, 1);

  driver.send_status = Status::Ok();
  auto complete = server.advance(
      401, handler, coordinator, shutdown, termination);
  ASSERT_TRUE(complete.ok());
  EXPECT_TRUE(*complete);
  EXPECT_EQ(domain_driver.kills, 1);
  EXPECT_EQ(driver.polls, 1);
  EXPECT_EQ(driver.sends, 2);
  auto ack = decode_engine_supervisor_shutdown_ack(driver.ack);
  ASSERT_TRUE(ack.ok());
  EXPECT_EQ(ack->disposition,
            EngineSupervisorShutdownAckDisposition::kAccepted);
}

TEST(EngineSupervisorShutdownServerTest,
     RetriesTransientDomainKillAgainstRetainedRequest) {
  ServerDriver driver;
  driver.request = request_frame();
  auto server = EngineSupervisorShutdownServer::Create(driver).value();
  auto handler = EngineSupervisorShutdownHandler::Create(3, 9).value();
  auto coordinator = ready_coordinator();
  auto shutdown = EngineShutdownController::Create({50, 25}).value();
  DomainDriver domain_driver;
  domain_driver.kill_status = Status::Unavailable("injected kill failure");
  auto termination =
      EngineGenerationDomainTermination::Create(domain_driver).value();

  auto failed = server.advance(
      400, handler, coordinator, shutdown, termination);
  ASSERT_TRUE(failed.ok());
  EXPECT_FALSE(*failed);
  EXPECT_EQ(domain_driver.kills, 1);
  EXPECT_EQ(driver.sends, 0);

  domain_driver.kill_status = Status::Ok();
  auto complete = server.advance(
      401, handler, coordinator, shutdown, termination);
  ASSERT_TRUE(complete.ok());
  EXPECT_TRUE(*complete);
  EXPECT_EQ(domain_driver.kills, 2);
  EXPECT_EQ(driver.polls, 1);
  EXPECT_EQ(driver.sends, 1);
}

TEST(EngineSupervisorShutdownServerTest,
     InvalidFramePoisonsServerWithoutShutdownMutation) {
  ServerDriver driver;
  driver.request = std::vector<std::byte>(3);
  auto server = EngineSupervisorShutdownServer::Create(driver).value();
  auto handler = EngineSupervisorShutdownHandler::Create(3, 9).value();
  auto coordinator = ready_coordinator();
  auto shutdown = EngineShutdownController::Create({50, 25}).value();
  DomainDriver domain_driver;
  auto termination =
      EngineGenerationDomainTermination::Create(domain_driver).value();

  EXPECT_FALSE(server.advance(
      400, handler, coordinator, shutdown, termination).ok());
  EXPECT_FALSE(server.advance(
      401, handler, coordinator, shutdown, termination).ok());
  EXPECT_EQ(domain_driver.kills, 0);
  EXPECT_EQ(coordinator.lifecycle_state(), EngineGenerationState::kReady);
}

}  // namespace
}  // namespace pih
