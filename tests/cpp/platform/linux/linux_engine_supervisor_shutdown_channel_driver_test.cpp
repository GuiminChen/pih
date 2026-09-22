#include "pih/platform/linux/linux_engine_supervisor_shutdown_channel_driver.h"

#include <array>
#include <utility>

#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

namespace pih {
namespace {

class SocketPair final {
 public:
  explicit SocketPair(int type) {
    if (::socketpair(AF_UNIX, type | SOCK_CLOEXEC, 0, descriptors_) != 0) {
      descriptors_[0] = descriptors_[1] = -1;
    }
  }
  ~SocketPair() {
    for (const int descriptor : descriptors_) {
      if (descriptor >= 0) (void)::close(descriptor);
    }
  }
  int release_first() {
    const int value = descriptors_[0];
    descriptors_[0] = -1;
    return value;
  }
  int first() const { return descriptors_[0]; }
  int second() const { return descriptors_[1]; }

 private:
  int descriptors_[2]{-1, -1};
};

TEST(LinuxEngineSupervisorShutdownChannelDriverTest,
     DuplicatesValidatedSeqpacketAndTransfersExactFrames) {
  SocketPair sockets(SOCK_SEQPACKET);
  ASSERT_GE(sockets.first(), 0);
  auto value = LinuxEngineSupervisorShutdownChannelDriver::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()));
  ASSERT_TRUE(value.ok());
  auto driver = std::move(*value);
  const int inherited = sockets.release_first();
  ASSERT_EQ(::close(inherited), 0);

  const auto request = encode_engine_supervisor_shutdown_request(
      {7, 11, 1, 900, EngineSupervisorShutdownRequestKind::kForceStop});
  ASSERT_TRUE(driver.send_request(request).ok());
  std::array<std::byte, kEngineSupervisorShutdownFrameBytes> received{};
  ASSERT_EQ(::recv(sockets.second(), received.data(), received.size(), 0),
            static_cast<ssize_t>(received.size()));
  EXPECT_EQ(received, request);

  const auto ack = encode_engine_supervisor_shutdown_ack(
      {7, 11, 1, 850,
       EngineSupervisorShutdownAckDisposition::kAccepted});
  ASSERT_EQ(::send(sockets.second(), ack.data(), ack.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(ack.size()));
  const auto polled = driver.poll_ack();
  ASSERT_TRUE(polled.ok());
  ASSERT_TRUE(polled->has_value());
  EXPECT_EQ(polled->value(), std::vector<std::byte>(ack.begin(), ack.end()));
}

TEST(LinuxEngineSupervisorShutdownChannelDriverTest,
     RejectsWrongSocketAndPeerIdentity) {
  SocketPair stream(SOCK_STREAM);
  EXPECT_FALSE(LinuxEngineSupervisorShutdownChannelDriver::Create(
      stream.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid())).ok());
  SocketPair packet(SOCK_SEQPACKET);
  EXPECT_FALSE(LinuxEngineSupervisorShutdownChannelDriver::Create(
      packet.first(), static_cast<std::uint64_t>(::getpid()) + 1,
      static_cast<std::uint32_t>(::getuid())).ok());
}

TEST(LinuxEngineSupervisorShutdownChannelDriverTest,
     RejectsTruncatedOversizedAcknowledgement) {
  SocketPair sockets(SOCK_SEQPACKET);
  auto value = LinuxEngineSupervisorShutdownChannelDriver::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()));
  ASSERT_TRUE(value.ok());
  auto driver = std::move(*value);
  std::array<std::byte, kEngineSupervisorShutdownFrameBytes + 1> oversized{};
  ASSERT_EQ(::send(sockets.second(), oversized.data(), oversized.size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(oversized.size()));
  EXPECT_FALSE(driver.poll_ack().ok());
}

TEST(LinuxEngineSupervisorShutdownClientTest,
     OwnsDriverAndCompletesOneGenerationBoundTransaction) {
  SocketPair sockets(SOCK_SEQPACKET);
  auto value = LinuxEngineSupervisorShutdownClient::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()), 7, 11, 900);
  ASSERT_TRUE(value.ok());
  auto client = std::move(*value);
  EXPECT_EQ(client->engine_generation(), 7U);
  EXPECT_EQ(client->engine_epoch(), 11U);

  auto pending = client->advance(800);
  ASSERT_TRUE(pending.ok());
  EXPECT_FALSE(*pending);
  std::array<std::byte, kEngineSupervisorShutdownFrameBytes> request_bytes{};
  ASSERT_EQ(::recv(sockets.second(), request_bytes.data(), request_bytes.size(),
                   0),
            static_cast<ssize_t>(request_bytes.size()));
  const auto request = decode_engine_supervisor_shutdown_request(request_bytes);
  ASSERT_TRUE(request.ok());
  EXPECT_EQ(request->engine_generation, 7U);
  EXPECT_EQ(request->engine_epoch, 11U);
  EXPECT_EQ(request->request_identity, 1U);
  EXPECT_EQ(request->deadline_ns, 900U);
  EXPECT_EQ(request->kind, EngineSupervisorShutdownRequestKind::kForceStop);

  const auto ack = encode_engine_supervisor_shutdown_ack(
      {7, 11, 1, 850, EngineSupervisorShutdownAckDisposition::kAccepted});
  ASSERT_EQ(::send(sockets.second(), ack.data(), ack.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(ack.size()));
  auto complete = client->advance(860);
  ASSERT_TRUE(complete.ok());
  EXPECT_TRUE(*complete);
  EXPECT_TRUE(client->request_acknowledged());
}

TEST(LinuxEngineSupervisorShutdownClientTest, RejectsInvalidTransactionBounds) {
  SocketPair sockets(SOCK_SEQPACKET);
  EXPECT_FALSE(LinuxEngineSupervisorShutdownClient::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()), 0, 11, 900).ok());
  EXPECT_FALSE(LinuxEngineSupervisorShutdownClient::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()), 7, 0, 900).ok());
  EXPECT_FALSE(LinuxEngineSupervisorShutdownClient::Create(
      sockets.first(), static_cast<std::uint64_t>(::getpid()),
      static_cast<std::uint32_t>(::getuid()), 7, 11, 0).ok());
}

}  // namespace
}  // namespace pih
