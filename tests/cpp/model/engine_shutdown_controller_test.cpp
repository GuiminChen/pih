#include "pih/model/engine_shutdown_controller.h"

#include <gtest/gtest.h>

namespace pih {

TEST(EngineShutdownControllerTest, DrainsThenStopsWithinAbsoluteDeadlines) {
  auto value = EngineShutdownController::Create({10, 5}).value();
  ASSERT_EQ(*value.request_termination(100),
            EngineShutdownAction::kWithdrawReadiness);
  EXPECT_EQ(value.state(), EngineShutdownState::kDraining);
  EXPECT_EQ(value.drain_deadline_ns(), 110U);
  EXPECT_EQ(value.force_deadline_ns(), 115U);
  EXPECT_EQ(*value.poll(109, true), EngineShutdownAction::kNone);
  EXPECT_EQ(*value.poll(110, true), EngineShutdownAction::kBeginStop);
  EXPECT_EQ(value.state(), EngineShutdownState::kStopping);
  EXPECT_EQ(*value.poll(114, false), EngineShutdownAction::kNone);
  EXPECT_EQ(*value.poll(115, false),
            EngineShutdownAction::kForceKillDomain);
  EXPECT_EQ(value.state(), EngineShutdownState::kForceStopping);
  EXPECT_EQ(*value.poll(116, false), EngineShutdownAction::kNone);
  ASSERT_TRUE(value.mark_domain_empty().ok());
  EXPECT_EQ(value.state(), EngineShutdownState::kComplete);
}

TEST(EngineShutdownControllerTest, CompletedDrainBeginsStopEarly) {
  auto value = EngineShutdownController::Create({10, 5}).value();
  ASSERT_TRUE(value.request_termination(100).ok());
  EXPECT_EQ(*value.poll(101, false), EngineShutdownAction::kBeginStop);
  EXPECT_EQ(value.state(), EngineShutdownState::kStopping);
}

TEST(EngineShutdownControllerTest, RepeatedTerminationForcesExactlyOnce) {
  auto value = EngineShutdownController::Create({10, 5}).value();
  ASSERT_TRUE(value.request_termination(100).ok());
  EXPECT_EQ(*value.request_termination(101),
            EngineShutdownAction::kForceKillDomain);
  EXPECT_EQ(value.state(), EngineShutdownState::kForceStopping);
  EXPECT_EQ(*value.request_termination(102), EngineShutdownAction::kNone);
  EXPECT_EQ(*value.poll(103, true), EngineShutdownAction::kNone);
}

TEST(EngineShutdownControllerTest, PollJumpToForceDeadlineKillsImmediately) {
  auto value = EngineShutdownController::Create({10, 5}).value();
  ASSERT_TRUE(value.request_termination(100).ok());
  EXPECT_EQ(*value.poll(109, true), EngineShutdownAction::kNone);
  EXPECT_EQ(*value.poll(115, true),
            EngineShutdownAction::kForceKillDomain);
  EXPECT_EQ(value.state(), EngineShutdownState::kForceStopping);
}

TEST(EngineShutdownControllerTest, StartupAbortSkipsDrainButRemainsBounded) {
  auto value = EngineShutdownController::Create({10, 5}).value();
  ASSERT_EQ(*value.abort_start(100), EngineShutdownAction::kBeginStop);
  EXPECT_EQ(value.state(), EngineShutdownState::kStopping);
  EXPECT_EQ(value.drain_deadline_ns(), 100U);
  EXPECT_EQ(value.force_deadline_ns(), 105U);
  EXPECT_EQ(*value.poll(104, false), EngineShutdownAction::kNone);
  EXPECT_EQ(*value.poll(105, false),
            EngineShutdownAction::kForceKillDomain);
}

TEST(EngineShutdownControllerTest, RejectsInvalidPolicyClockAndCompletion) {
  EXPECT_FALSE(EngineShutdownController::Create({0, 5}).ok());
  EXPECT_FALSE(EngineShutdownController::Create({10, 0}).ok());
  auto overflow = EngineShutdownController::Create({10, 5}).value();
  EXPECT_FALSE(overflow.request_termination(UINT64_MAX - 14).ok());
  auto value = EngineShutdownController::Create({10, 5}).value();
  EXPECT_FALSE(value.mark_domain_empty().ok());
  ASSERT_TRUE(value.request_termination(100).ok());
  EXPECT_FALSE(value.poll(99, true).ok());
}

}  // namespace pih
