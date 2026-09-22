#include "pih/model/bound_engine_inet_diag_transport.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class InetDiagBackend final : public EngineInetDiagDescriptorBackend {
 public:
  Result<EngineInetDiagNamespaceHandleState> namespace_state(
      std::int32_t namespace_descriptor,
      std::int32_t netlink_descriptor) override {
    ++state_calls;
    namespace_fd = namespace_descriptor;
    netlink_fd = netlink_descriptor;
    return state;
  }
  Result<EngineInetDiagDumpTicket> begin_dump(
      std::int32_t netlink_descriptor,
      std::uint64_t maximum_duration_ms) override {
    ++begin_calls;
    netlink_fd = netlink_descriptor;
    duration_ms = maximum_duration_ms;
    return ticket;
  }
  Result<std::optional<EngineInetDiagMultipartMessage>> receive(
      std::int32_t netlink_descriptor,
      const EngineInetDiagDumpTicket& value) override {
    ++receive_calls;
    netlink_fd = netlink_descriptor;
    received_ticket = value;
    return message;
  }

  int state_calls = 0;
  int begin_calls = 0;
  int receive_calls = 0;
  std::int32_t namespace_fd = -1;
  std::int32_t netlink_fd = -1;
  std::uint64_t duration_ms = 0;
  EngineInetDiagNamespaceHandleState state =
      EngineInetDiagNamespaceHandleState::kPresent;
  EngineInetDiagDumpTicket ticket{7, 1000};
  EngineInetDiagDumpTicket received_ticket{};
  std::optional<EngineInetDiagMultipartMessage> message{
      EngineInetDiagMultipartMessage{
          7, EngineInetDiagMultipartMessageKind::kDone, true, false, 0, {}}};
};

TEST(BoundEngineInetDiagTransportTest, RoutesAllOperationsToBoundDescriptors) {
  InetDiagBackend backend;
  auto transport = BoundEngineInetDiagTransport::Create(
      {50, 5, 6}, backend).value();
  EXPECT_EQ(*transport.namespace_state(50),
            EngineInetDiagNamespaceHandleState::kPresent);
  EXPECT_TRUE(transport.begin_dump(50, 30000).ok());
  EXPECT_TRUE(transport.receive({7, 1000}).ok());
  EXPECT_EQ(backend.namespace_fd, 5);
  EXPECT_EQ(backend.netlink_fd, 6);
  EXPECT_EQ(backend.duration_ms, 30000U);
  EXPECT_EQ(backend.received_ticket.sequence, 7U);
}

TEST(BoundEngineInetDiagTransportTest, RejectsWrongNamespaceBeforeBackend) {
  InetDiagBackend backend;
  auto transport = BoundEngineInetDiagTransport::Create(
      {50, 5, 6}, backend).value();
  EXPECT_FALSE(transport.namespace_state(51).ok());
  EXPECT_FALSE(transport.begin_dump(51, 30000).ok());
  EXPECT_EQ(backend.state_calls, 0);
  EXPECT_EQ(backend.begin_calls, 0);
}

TEST(BoundEngineInetDiagTransportTest, PreservesBackendFailures) {
  InetDiagBackend backend;
  auto transport = BoundEngineInetDiagTransport::Create(
      {50, 5, 6}, backend).value();
  backend.state = static_cast<EngineInetDiagNamespaceHandleState>(2);
  EXPECT_EQ(static_cast<int>(*transport.namespace_state(50)), 2);
  backend.ticket = {};
  EXPECT_EQ(transport.begin_dump(50, 30000)->sequence, 0U);
}

TEST(BoundEngineInetDiagTransportTest, RejectsInvalidBinding) {
  InetDiagBackend backend;
  EXPECT_FALSE(BoundEngineInetDiagTransport::Create({0, 5, 6}, backend).ok());
  EXPECT_FALSE(BoundEngineInetDiagTransport::Create({50, -1, 6}, backend).ok());
  EXPECT_FALSE(BoundEngineInetDiagTransport::Create({50, 5, -1}, backend).ok());
  EXPECT_FALSE(BoundEngineInetDiagTransport::Create({50, 5, 5}, backend).ok());
}

}  // namespace
}  // namespace pih
