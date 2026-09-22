#include "pih/model/deepseek_request_lifecycle.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekRequestLifecycleTest, CancelBeforeCommitReleasesImmediately) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  EXPECT_TRUE(request.admit().ok());
  EXPECT_FALSE(request.validate_prepare(0).ok());
  EXPECT_TRUE(request.validate_prepare(9).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kAdmitted);
  EXPECT_TRUE(request.prepare(9).ok());
  EXPECT_FALSE(request.validate_cancel(8).ok());
  EXPECT_TRUE(request.validate_cancel(7).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kPrepared);
  EXPECT_TRUE(request.cancel(7).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kCancelled);
  EXPECT_TRUE(request.user_terminal());
  EXPECT_TRUE(request.backend_drained());
  EXPECT_TRUE(request.suppress_output());
  EXPECT_FALSE(request.commit(9).ok());
}

TEST(DeepSeekRequestLifecycleTest, CancelAfterCommitDrainsMatchingPlan) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  ASSERT_TRUE(request.admit().ok());
  ASSERT_TRUE(request.prepare(9).ok());
  ASSERT_TRUE(request.commit(9).ok());
  EXPECT_TRUE(request.cancel(7).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kDraining);
  EXPECT_TRUE(request.user_terminal());
  EXPECT_FALSE(request.backend_drained());
  EXPECT_TRUE(request.suppress_output());
  EXPECT_TRUE(request.cancel(7).ok());
  EXPECT_FALSE(request.backend_complete(8).ok());
  EXPECT_TRUE(request.backend_complete(9).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kCancelled);
  EXPECT_TRUE(request.backend_drained());
}

TEST(DeepSeekRequestLifecycleTest, GenerationPreventsIdReuseCancellation) {
  auto request = DeepSeekRequestLifecycle::Create(41, 8).value();
  EXPECT_FALSE(request.cancel(7).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kSubmitted);
  EXPECT_FALSE(DeepSeekRequestLifecycle::Create(0, 1).ok());
  EXPECT_FALSE(DeepSeekRequestLifecycle::Create(1, 0).ok());
}

TEST(DeepSeekRequestLifecycleTest, NormalCompletionPublishesOnce) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  ASSERT_TRUE(request.admit().ok());
  ASSERT_TRUE(request.prepare(9).ok());
  ASSERT_TRUE(request.commit(9).ok());
  EXPECT_TRUE(request.backend_complete(9).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kCompleted);
  EXPECT_TRUE(request.user_terminal());
  EXPECT_TRUE(request.backend_drained());
  EXPECT_FALSE(request.cancel(7).ok());
}

TEST(DeepSeekRequestLifecycleTest, CompletionValidationDoesNotDrainRequest) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  ASSERT_TRUE(request.admit().ok());
  ASSERT_TRUE(request.prepare(9).ok());
  ASSERT_TRUE(request.commit(9).ok());
  EXPECT_FALSE(request.validate_backend_complete(8).ok());
  EXPECT_TRUE(request.validate_backend_complete(9).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kCommitted);
  EXPECT_FALSE(request.backend_drained());
}

TEST(DeepSeekRequestLifecycleTest, FailureAfterCommitDrainsToFailed) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  ASSERT_TRUE(request.admit().ok());
  ASSERT_TRUE(request.prepare(9).ok());
  ASSERT_TRUE(request.commit(9).ok());
  EXPECT_TRUE(request.fail().ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kDraining);
  EXPECT_TRUE(request.user_terminal());
  EXPECT_FALSE(request.backend_drained());
  EXPECT_TRUE(request.backend_complete(9).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kFailed);
  EXPECT_TRUE(request.backend_drained());
}

TEST(DeepSeekRequestLifecycleTest, AbortPrepareReturnsToAdmitted) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  ASSERT_TRUE(request.admit().ok());
  ASSERT_TRUE(request.prepare(9).ok());
  EXPECT_FALSE(request.validate_abort_prepare(8).ok());
  EXPECT_TRUE(request.validate_abort_prepare(9).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kPrepared);
  EXPECT_FALSE(request.abort_prepare(8).ok());
  EXPECT_TRUE(request.abort_prepare(9).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kAdmitted);
  EXPECT_TRUE(request.prepare(10).ok());
}

TEST(DeepSeekRequestLifecycleTest, CommitValidationDoesNotMutatePreparedState) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  ASSERT_TRUE(request.admit().ok());
  ASSERT_TRUE(request.prepare(9).ok());
  EXPECT_FALSE(request.validate_commit(8).ok());
  EXPECT_TRUE(request.validate_commit(9).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kPrepared);
  EXPECT_TRUE(request.abort_prepare(9).ok());
}

TEST(DeepSeekRequestLifecycleTest, NonTerminalPlanReturnsToAdmitted) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  ASSERT_TRUE(request.admit().ok());
  ASSERT_TRUE(request.prepare(9).ok());
  ASSERT_TRUE(request.commit(9).ok());
  ASSERT_TRUE(request.backend_complete(9, false).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kAdmitted);
  EXPECT_FALSE(request.user_terminal());
  EXPECT_FALSE(request.backend_drained());
  EXPECT_TRUE(request.prepare(10).ok());
}

TEST(DeepSeekRequestLifecycleTest, CancelDrainOverridesNonTerminalPlan) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  ASSERT_TRUE(request.admit().ok());
  ASSERT_TRUE(request.prepare(9).ok());
  ASSERT_TRUE(request.commit(9).ok());
  ASSERT_TRUE(request.cancel(7).ok());
  ASSERT_TRUE(request.backend_complete(9, false).ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kCancelled);
  EXPECT_TRUE(request.user_terminal());
  EXPECT_TRUE(request.backend_drained());
}

TEST(DeepSeekRequestLifecycleTest, FinishAfterNonTerminalPlanPublishesTerminal) {
  auto request = DeepSeekRequestLifecycle::Create(41, 7).value();
  ASSERT_TRUE(request.admit().ok());
  ASSERT_TRUE(request.prepare(9).ok());
  ASSERT_TRUE(request.commit(9).ok());
  ASSERT_TRUE(request.backend_complete(9, false).ok());
  ASSERT_TRUE(request.finish().ok());
  EXPECT_EQ(request.state(), DeepSeekRequestState::kCompleted);
  EXPECT_TRUE(request.user_terminal());
  EXPECT_TRUE(request.backend_drained());
  EXPECT_FALSE(request.finish().ok());
}

} }  // namespace pih
