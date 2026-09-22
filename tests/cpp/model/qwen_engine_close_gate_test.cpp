#include "pih/model/qwen_engine_close_gate.h"

#include <chrono>
#include <future>
#include <optional>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenEngineCloseGateTest, SealsAdmissionBeforeCloseWorkBegins) {
  QwenEngineCloseGate gate;
  EXPECT_TRUE(gate.require_open().ok());

  EXPECT_TRUE(gate.request_close());
  EXPECT_EQ(gate.state(), QwenEngineCloseState::kClosing);
  EXPECT_EQ(gate.require_open().code(), StatusCode::kFailedPrecondition);
  EXPECT_FALSE(gate.request_close());

  gate.finish_close(Status::Ok());
  EXPECT_EQ(gate.state(), QwenEngineCloseState::kClosed);
  EXPECT_EQ(gate.require_open().code(), StatusCode::kFailedPrecondition);
  EXPECT_FALSE(gate.request_close());
}

TEST(QwenEngineCloseGateTest, FinishCannotCloseAnOpenEpoch) {
  QwenEngineCloseGate gate;
  gate.finish_close(Status::Ok());
  EXPECT_EQ(gate.state(), QwenEngineCloseState::kOpen);
  EXPECT_TRUE(gate.require_open().ok());
}

TEST(QwenEngineCloseGateTest, ConcurrentCloserWaitsForOwnerCompletion) {
  QwenEngineCloseGate gate;
  ASSERT_TRUE(gate.request_close());
  auto waiter = std::async(std::launch::async, [&gate] {
    const Status outcome = gate.wait_closed();
    return std::pair{gate.state(), outcome.code()};
  });
  EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(10)),
            std::future_status::timeout);
  gate.finish_close(Status::Internal("injected close failure"));
  EXPECT_EQ(waiter.get(),
            (std::pair{QwenEngineCloseState::kClosed, StatusCode::kInternal}));
}

TEST(QwenEngineCloseGateTest, CloseWaitsForEveryAdmittedOperationLease) {
  QwenEngineCloseGate gate;
  auto first = gate.enter();
  auto second = gate.enter();
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  std::optional<QwenEngineOperationLease> first_lease(std::move(*first));
  std::optional<QwenEngineOperationLease> second_lease(std::move(*second));
  ASSERT_TRUE(gate.request_close());
  EXPECT_EQ(gate.enter().status().code(), StatusCode::kFailedPrecondition);

  auto drained = std::async(std::launch::async, [&gate] {
    gate.wait_drained();
    return true;
  });
  EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(10)),
            std::future_status::timeout);
  first_lease.reset();
  EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(10)),
            std::future_status::timeout);
  second_lease.reset();
  EXPECT_TRUE(drained.get());
  gate.finish_close(Status::Ok());
}

TEST(QwenEngineCloseGateTest, MovingLeaseTransfersExactlyOneRelease) {
  QwenEngineCloseGate gate;
  auto entered = gate.enter();
  ASSERT_TRUE(entered.ok());
  std::future<bool> drained;
  {
    QwenEngineOperationLease moved(std::move(*entered));
    ASSERT_TRUE(gate.request_close());
    drained = std::async(std::launch::async, [&gate] {
      gate.wait_drained();
      return true;
    });
    EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(10)),
              std::future_status::timeout);
  }
  EXPECT_TRUE(drained.get());
  gate.finish_close(Status::Ok());
}

TEST(QwenEngineCloseGateTest, EveryConcurrentCloserObservesSameFailure) {
  QwenEngineCloseGate gate;
  ASSERT_TRUE(gate.request_close());
  auto first = std::async(std::launch::async, [&gate] {
    return gate.wait_closed();
  });
  auto second = std::async(std::launch::async, [&gate] {
    return gate.wait_closed();
  });
  gate.finish_close(Status::FailedPrecondition("undrained packed request"));
  EXPECT_EQ(first.get().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(second.get().code(), StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
