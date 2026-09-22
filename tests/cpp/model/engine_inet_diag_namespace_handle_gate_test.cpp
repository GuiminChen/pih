#include "pih/model/engine_inet_diag_namespace_handle_gate.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class NamespaceProbe final : public EngineInetDiagNamespaceHandleProbe {
 public:
  Result<EngineInetDiagNamespaceHandleObservation> inspect(
      std::int32_t namespace_descriptor,
      std::int32_t netlink_descriptor) override {
    ++calls;
    namespace_fd = namespace_descriptor;
    netlink_fd = netlink_descriptor;
    return result;
  }
  int calls = 0;
  std::int32_t namespace_fd = -1;
  std::int32_t netlink_fd = -1;
  Result<EngineInetDiagNamespaceHandleObservation> result =
      EngineInetDiagNamespaceHandleObservation{true, true, true, true};
};

TEST(EngineInetDiagNamespaceHandleGateTest, AcceptsMatchingOpenHandles) {
  NamespaceProbe probe;
  auto gate = EngineInetDiagNamespaceHandleGate::Create(5, 6, probe).value();
  auto state = gate.inspect();
  ASSERT_TRUE(state.ok());
  EXPECT_EQ(*state, EngineInetDiagNamespaceHandleState::kPresent);
  EXPECT_EQ(probe.namespace_fd, 5);
  EXPECT_EQ(probe.netlink_fd, 6);
}

TEST(EngineInetDiagNamespaceHandleGateTest, ProvesOnlyJointDestruction) {
  NamespaceProbe probe;
  probe.result = EngineInetDiagNamespaceHandleObservation{};
  auto gate = EngineInetDiagNamespaceHandleGate::Create(5, 6, probe).value();
  auto state = gate.inspect();
  ASSERT_TRUE(state.ok());
  EXPECT_EQ(*state, EngineInetDiagNamespaceHandleState::kDestroyed);
}

TEST(EngineInetDiagNamespaceHandleGateTest, RejectsPartialTeardown) {
  for (int mutation = 0; mutation < 2; ++mutation) {
    NamespaceProbe probe;
    probe.result = mutation == 0
        ? EngineInetDiagNamespaceHandleObservation{true, true, false, false}
        : EngineInetDiagNamespaceHandleObservation{false, false, true, true};
    auto gate = EngineInetDiagNamespaceHandleGate::Create(5, 6, probe).value();
    auto state = gate.inspect();
    ASSERT_FALSE(state.ok());
    EXPECT_EQ(state.status().code(), StatusCode::kFailedPrecondition);
  }
}

TEST(EngineInetDiagNamespaceHandleGateTest, RejectsWrongHandleTypes) {
  for (int mutation = 0; mutation < 2; ++mutation) {
    NamespaceProbe probe;
    if (mutation == 0) probe.result->namespace_is_network_namespace = false;
    if (mutation == 1) probe.result->netlink_is_sock_diag = false;
    auto gate = EngineInetDiagNamespaceHandleGate::Create(5, 6, probe).value();
    EXPECT_FALSE(gate.inspect().ok()) << mutation;
  }
}

TEST(EngineInetDiagNamespaceHandleGateTest, PreservesUnavailableAndRejectsFds) {
  NamespaceProbe probe;
  probe.result = Status::Unavailable("handle visibility unavailable");
  auto gate = EngineInetDiagNamespaceHandleGate::Create(5, 6, probe).value();
  EXPECT_EQ(gate.inspect().status().code(), StatusCode::kUnavailable);
  EXPECT_FALSE(EngineInetDiagNamespaceHandleGate::Create(-1, 6, probe).ok());
  EXPECT_FALSE(EngineInetDiagNamespaceHandleGate::Create(5, -1, probe).ok());
  EXPECT_FALSE(EngineInetDiagNamespaceHandleGate::Create(5, 5, probe).ok());
}

}  // namespace
}  // namespace pih
