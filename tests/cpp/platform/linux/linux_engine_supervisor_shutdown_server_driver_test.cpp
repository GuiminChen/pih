#include "pih/platform/linux/linux_engine_supervisor_shutdown_server_driver.h"

#include <array>
#include <utility>

#include <gtest/gtest.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace pih {
namespace {

class ServerSocketPair final {
 public:
  explicit ServerSocketPair(int type) {
    if (::socketpair(AF_UNIX, type | SOCK_CLOEXEC, 0, descriptors_) != 0) {
      descriptors_[0] = descriptors_[1] = -1;
    }
  }
  ~ServerSocketPair() {
    for (const int descriptor : descriptors_) {
      if (descriptor >= 0) (void)::close(descriptor);
    }
  }
  int first() const { return descriptors_[0]; }
  int second() const { return descriptors_[1]; }

 private:
  int descriptors_[2]{-1, -1};
};

Sha256Digest service_digest() {
  Sha256Digest value{};
  value.bytes.fill(std::byte{8});
  return value;
}

EngineSupervisionCoordinator service_coordinator() {
  const EngineAllocationLeaseExpectation expectation{service_digest(), 17, 19};
  const EngineAllocationLeaseObservation observation{
      service_digest(), 17, 19, true, true};
  auto coordinator = EngineSupervisionCoordinator::Create(
      7, 1, 100, {10, 100, 1'000, 3}, expectation).value();
  EXPECT_TRUE(coordinator.begin_start(observation, 100).ok());
  EXPECT_TRUE(coordinator.publish_ready(observation).ok());
  return coordinator;
}

class ServiceDomainDriver final : public EngineGenerationDomainDriver {
 public:
  Status force_kill_domain() override {
    ++kills;
    return Status::Ok();
  }
  Result<bool> domain_empty() override { return false; }
  int kills = 0;
};

TEST(LinuxEngineSupervisorShutdownServerDriverTest,
     ValidatesEnginePeerAndTransfersExactFrames) {
  ServerSocketPair sockets(SOCK_SEQPACKET);
  ASSERT_GE(sockets.first(), 0);
  auto value = LinuxEngineSupervisorShutdownServerDriver::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()));
  ASSERT_TRUE(value.ok());
  auto driver = std::move(*value);

  auto empty = driver.poll_request();
  ASSERT_TRUE(empty.ok());
  EXPECT_FALSE(empty->has_value());

  const auto request = encode_engine_supervisor_shutdown_request(
      {7, 11, 1, 900, EngineSupervisorShutdownRequestKind::kForceStop});
  ASSERT_EQ(::send(sockets.second(), request.data(), request.size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(request.size()));
  auto received = driver.poll_request();
  ASSERT_TRUE(received.ok());
  ASSERT_TRUE(received->has_value());
  EXPECT_EQ(received->value(),
            std::vector<std::byte>(request.begin(), request.end()));

  const auto ack = encode_engine_supervisor_shutdown_ack(
      {7, 11, 1, 850, EngineSupervisorShutdownAckDisposition::kAccepted});
  ASSERT_TRUE(driver.send_ack(ack).ok());
  std::array<std::byte, kEngineSupervisorShutdownFrameBytes> ack_received{};
  ASSERT_EQ(::recv(sockets.second(), ack_received.data(), ack_received.size(), 0),
            static_cast<ssize_t>(ack_received.size()));
  EXPECT_EQ(ack_received, ack);
}

TEST(LinuxEngineSupervisorShutdownServerDriverTest,
     RejectsWrongSocketPeerAndOversizedRequest) {
  ServerSocketPair stream(SOCK_STREAM);
  EXPECT_FALSE(LinuxEngineSupervisorShutdownServerDriver::Create(
      stream.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid())).ok());

  ServerSocketPair wrong_peer(SOCK_SEQPACKET);
  auto wrong_value = LinuxEngineSupervisorShutdownServerDriver::Create(
      wrong_peer.first(), static_cast<std::uint64_t>(::getpid()) + 1,
      static_cast<std::uint32_t>(::getuid()));
  ASSERT_TRUE(wrong_value.ok());
  auto wrong_driver = std::move(*wrong_value);
  const auto request = encode_engine_supervisor_shutdown_request(
      {7, 11, 1, 900, EngineSupervisorShutdownRequestKind::kForceStop});
  ASSERT_EQ(::send(wrong_peer.second(), request.data(), request.size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(request.size()));
  EXPECT_FALSE(wrong_driver.poll_request().ok());

  ServerSocketPair sockets(SOCK_SEQPACKET);
  auto value = LinuxEngineSupervisorShutdownServerDriver::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()));
  ASSERT_TRUE(value.ok());
  auto driver = std::move(*value);
  std::array<std::byte, kEngineSupervisorShutdownFrameBytes + 1> oversized{};
  ASSERT_EQ(::send(sockets.second(), oversized.data(), oversized.size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(oversized.size()));
  EXPECT_FALSE(driver.poll_request().ok());
}

TEST(LinuxEngineSupervisorShutdownServerDriverTest,
     UsesPerMessageCredentialsForPreforkInheritedSocketpair) {
  ServerSocketPair sockets(SOCK_SEQPACKET);
  const int pass_credentials = 1;
  ASSERT_EQ(::setsockopt(sockets.first(), SOL_SOCKET, SO_PASSCRED,
                         &pass_credentials, sizeof(pass_credentials)),
            0);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    const auto request = encode_engine_supervisor_shutdown_request(
        {7, 11, 1, 900, EngineSupervisorShutdownRequestKind::kForceStop});
    const auto sent = ::send(sockets.second(), request.data(), request.size(),
                             MSG_NOSIGNAL);
    ::_exit(sent == static_cast<ssize_t>(request.size()) ? 0 : 3);
  }

  auto value = LinuxEngineSupervisorShutdownServerDriver::Create(
      sockets.first(), static_cast<std::uint64_t>(child),
      static_cast<std::uint32_t>(::getuid()));
  ASSERT_TRUE(value.ok());
  auto driver = std::move(*value);

  struct pollfd readiness {
    sockets.first(), POLLIN, 0
  };
  ASSERT_EQ(::poll(&readiness, 1, 1'000), 1);
  auto received = driver.poll_request();
  ASSERT_TRUE(received.ok()) << received.status().message();
  ASSERT_TRUE(received->has_value());

  int child_status = 0;
  ASSERT_EQ(::waitpid(child, &child_status, 0), child);
  EXPECT_TRUE(WIFEXITED(child_status));
  EXPECT_EQ(WEXITSTATUS(child_status), 0);
}

TEST(LinuxEngineSupervisorShutdownServiceTest,
     OwnsGenerationBoundServerAndAcknowledgesExecutedKill) {
  ServerSocketPair sockets(SOCK_SEQPACKET);
  auto value = LinuxEngineSupervisorShutdownService::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()), 7, 11);
  ASSERT_TRUE(value.ok());
  auto service = std::move(*value);
  EXPECT_EQ(service->engine_generation(), 7U);
  EXPECT_EQ(service->engine_epoch(), 11U);

  auto coordinator = service_coordinator();
  auto shutdown = EngineShutdownController::Create({50, 25}).value();
  ServiceDomainDriver domain_driver;
  auto termination =
      EngineGenerationDomainTermination::Create(domain_driver).value();
  const auto request = encode_engine_supervisor_shutdown_request(
      {7, 11, 1, 900, EngineSupervisorShutdownRequestKind::kForceStop});
  ASSERT_EQ(::send(sockets.second(), request.data(), request.size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(request.size()));

  auto complete = service->advance(800, coordinator, shutdown, termination);
  ASSERT_TRUE(complete.ok()) << complete.status().message();
  EXPECT_TRUE(*complete);
  EXPECT_TRUE(service->request_acknowledged());
  EXPECT_EQ(domain_driver.kills, 1);

  std::array<std::byte, kEngineSupervisorShutdownFrameBytes> ack_frame{};
  ASSERT_EQ(::recv(sockets.second(), ack_frame.data(), ack_frame.size(), 0),
            static_cast<ssize_t>(ack_frame.size()));
  auto ack = decode_engine_supervisor_shutdown_ack(ack_frame);
  ASSERT_TRUE(ack.ok());
  EXPECT_EQ(ack->disposition,
            EngineSupervisorShutdownAckDisposition::kAccepted);
}

TEST(LinuxEngineSupervisorShutdownServiceTest, RejectsInvalidGenerationBounds) {
  ServerSocketPair sockets(SOCK_SEQPACKET);
  EXPECT_FALSE(LinuxEngineSupervisorShutdownService::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()), 0, 11).ok());
  EXPECT_FALSE(LinuxEngineSupervisorShutdownService::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()), 7, 0).ok());
}

}  // namespace
}  // namespace pih
