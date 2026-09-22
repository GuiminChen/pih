#include "pih/model/qwen3_bf16_request_gate.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenBf16RequestGateTest, RejectsConcurrentRequestWithoutStateDrift) {
  QwenBf16RequestGate gate;

  EXPECT_TRUE(gate.begin().ok());
  const Status concurrent = gate.begin();

  EXPECT_EQ(concurrent.code(), StatusCode::kResourceExhausted);
  EXPECT_EQ(gate.state(), QwenBf16RequestGateState::kRunning);
  EXPECT_TRUE(gate.finish(true).ok());
  EXPECT_EQ(gate.state(), QwenBf16RequestGateState::kReady);
}

TEST(QwenBf16RequestGateTest, FailurePoisonsUntilTerminalClose) {
  QwenBf16RequestGate gate;
  ASSERT_TRUE(gate.begin().ok());

  EXPECT_TRUE(gate.finish(false).ok());
  EXPECT_EQ(gate.state(), QwenBf16RequestGateState::kPoisoned);
  EXPECT_EQ(gate.begin().code(), StatusCode::kFailedPrecondition);
  EXPECT_TRUE(gate.close().ok());
  EXPECT_EQ(gate.state(), QwenBf16RequestGateState::kClosed);
  EXPECT_TRUE(gate.close().ok());
}

TEST(QwenBf16RequestGateTest, CloseCannotRaceRunningRequest) {
  QwenBf16RequestGate gate;
  ASSERT_TRUE(gate.begin().ok());

  EXPECT_EQ(gate.close().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(gate.state(), QwenBf16RequestGateState::kRunning);
  EXPECT_TRUE(gate.finish(true).ok());
}

}  // namespace
}  // namespace pih
