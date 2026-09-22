#include "pih/model/engine_inet_diag_transport_adapter.h"

#include <deque>
#include <gtest/gtest.h>

namespace pih {
namespace {

class InetDiagTransport final : public EngineInetDiagTransport {
 public:
  Result<EngineInetDiagNamespaceHandleState> namespace_state(
      std::uint64_t namespace_identity) override {
    ++state_calls;
    return state;
  }
  Result<EngineInetDiagDumpTicket> begin_dump(
      std::uint64_t namespace_identity,
      std::uint64_t maximum_duration_ms) override {
    ++begin_calls;
    duration_ms = maximum_duration_ms;
    return ticket;
  }
  Result<std::optional<EngineInetDiagMultipartMessage>> receive(
      const EngineInetDiagDumpTicket& value) override {
    ++receive_calls;
    last_ticket = value;
    if (!receive_status.ok()) return receive_status;
    if (messages.empty())
      return std::optional<EngineInetDiagMultipartMessage>{};
    auto message = messages.front();
    messages.pop_front();
    return std::optional<EngineInetDiagMultipartMessage>{message};
  }

  int state_calls = 0;
  int begin_calls = 0;
  int receive_calls = 0;
  std::uint64_t duration_ms = 0;
  EngineInetDiagNamespaceHandleState state =
      EngineInetDiagNamespaceHandleState::kPresent;
  EngineInetDiagDumpTicket ticket{7, 1000};
  EngineInetDiagDumpTicket last_ticket{};
  Status receive_status = Status::Ok();
  std::deque<EngineInetDiagMultipartMessage> messages{
      EngineInetDiagMultipartMessage{
          7, EngineInetDiagMultipartMessageKind::kRow, true, false, 0,
          {40000, EngineNetworkSocketLifecycle::kActive, true, 64}},
      EngineInetDiagMultipartMessage{
          7, EngineInetDiagMultipartMessageKind::kDone, true, false, 0, {}}};
};

TEST(EngineInetDiagTransportAdapterTest, CapturesFiniteCompleteDump) {
  InetDiagTransport transport;
  auto adapter = EngineInetDiagTransportAdapter::Create(50, 30000, transport)
                     .value();
  auto snapshot = adapter.capture(50);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_TRUE(snapshot->namespace_handle_present);
  ASSERT_EQ(snapshot->rows.size(), 1U);
  EXPECT_EQ(snapshot->rows[0].local_port, 40000);
  EXPECT_EQ(transport.duration_ms, 30000U);
  EXPECT_EQ(transport.last_ticket.sequence, 7U);
  EXPECT_EQ(transport.last_ticket.absolute_deadline_ns, 1000U);
}

TEST(EngineInetDiagTransportAdapterTest, PreservesDestroyedHandleProof) {
  InetDiagTransport transport;
  transport.state = EngineInetDiagNamespaceHandleState::kDestroyed;
  auto adapter = EngineInetDiagTransportAdapter::Create(50, 30000, transport)
                     .value();
  auto snapshot = adapter.capture(50);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_TRUE(snapshot->namespace_destroyed);
  EXPECT_TRUE(snapshot->rows.empty());
  EXPECT_EQ(transport.begin_calls, 0);
}

TEST(EngineInetDiagTransportAdapterTest, RejectsEofBeforeDone) {
  InetDiagTransport transport;
  transport.messages.pop_back();
  auto adapter = EngineInetDiagTransportAdapter::Create(50, 30000, transport)
                     .value();
  auto snapshot = adapter.capture(50);
  ASSERT_FALSE(snapshot.ok());
  EXPECT_EQ(snapshot.status().code(), StatusCode::kUnavailable);
}

TEST(EngineInetDiagTransportAdapterTest, PreservesDeadlineAndProtocolFailure) {
  for (int mutation = 0; mutation < 2; ++mutation) {
    InetDiagTransport transport;
    if (mutation == 0)
      transport.receive_status = Status::DeadlineExceeded("dump deadline");
    else
      transport.messages.front().sequence = 8;
    auto adapter = EngineInetDiagTransportAdapter::Create(50, 30000, transport)
                       .value();
    auto snapshot = adapter.capture(50);
    ASSERT_FALSE(snapshot.ok()) << mutation;
    EXPECT_EQ(snapshot.status().code(),
              mutation == 0 ? StatusCode::kDeadlineExceeded
                            : StatusCode::kFailedPrecondition);
  }
}

TEST(EngineInetDiagTransportAdapterTest, RejectsInvalidTicketAndConfiguration) {
  InetDiagTransport transport;
  transport.ticket.sequence = 0;
  auto adapter = EngineInetDiagTransportAdapter::Create(50, 30000, transport)
                     .value();
  EXPECT_FALSE(adapter.capture(50).ok());
  transport.ticket = {7, 0};
  EXPECT_FALSE(adapter.capture(50).ok());
  EXPECT_FALSE(EngineInetDiagTransportAdapter::Create(0, 30000, transport).ok());
  EXPECT_FALSE(EngineInetDiagTransportAdapter::Create(50, 0, transport).ok());
}

}  // namespace
}  // namespace pih
