#include "pih/model/deepseek_boundary_credit_tracker.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(DeepSeekBoundaryCreditTrackerTest,
     ReservesExactlyTwoCreditsAndReusesWithNewGeneration) {
  auto tracker = DeepSeekBoundaryCreditTracker::Create(2);
  ASSERT_TRUE(tracker.ok()) << tracker.status().message();
  auto first = tracker->reserve(7, 11);
  auto second = tracker->reserve(8, 12);
  ASSERT_TRUE(first.ok()); ASSERT_TRUE(second.ok());
  EXPECT_NE(first->credit_index, second->credit_index);
  EXPECT_FALSE(tracker->reserve(9, 13).ok());

  ASSERT_TRUE(tracker->abort_prepare(*first).ok());
  auto reused = tracker->reserve(9, 13);
  ASSERT_TRUE(reused.ok());
  EXPECT_EQ(reused->credit_index, first->credit_index);
  EXPECT_GT(reused->credit_generation, first->credit_generation);
  EXPECT_FALSE(tracker->commit(*first).ok());
}

TEST(DeepSeekBoundaryCreditTrackerTest,
     CommittedCreditRequiresVerifiedCompletionBeforeRelease) {
  auto tracker = DeepSeekBoundaryCreditTracker::Create(2).value();
  auto handle = tracker.reserve(7, 11).value();
  ASSERT_TRUE(tracker.commit(handle).ok());
  EXPECT_FALSE(tracker.abort_prepare(handle).ok());
  EXPECT_FALSE(tracker.release(handle).ok());
  ASSERT_TRUE(tracker.complete_verified(handle).ok());
  ASSERT_TRUE(tracker.release(handle).ok());
  EXPECT_EQ(tracker.state(handle.credit_index),
            DeepSeekBoundaryCreditState::kFree);
}

TEST(DeepSeekBoundaryCreditTrackerTest,
     CommittedFailureQuarantinesCreditAndPoisonsTracker) {
  auto tracker = DeepSeekBoundaryCreditTracker::Create(2).value();
  auto handle = tracker.reserve(7, 11).value();
  ASSERT_TRUE(tracker.commit(handle).ok());
  ASSERT_TRUE(tracker.mark_suspect(handle).ok());
  EXPECT_TRUE(tracker.poisoned());
  EXPECT_EQ(tracker.state(handle.credit_index),
            DeepSeekBoundaryCreditState::kSuspect);
  EXPECT_FALSE(tracker.release(handle).ok());
  EXPECT_FALSE(tracker.reserve(8, 12).ok());
}

TEST(DeepSeekBoundaryCreditTrackerTest,
     RejectsDuplicateIdentityAndGenerationOverflow) {
  auto tracker = DeepSeekBoundaryCreditTracker::Create(2).value();
  auto handle = tracker.reserve(7, 11);
  ASSERT_TRUE(handle.ok());
  EXPECT_FALSE(tracker.reserve(7, 11).ok());
  EXPECT_FALSE(DeepSeekBoundaryCreditTracker::Create(0).ok());
  EXPECT_FALSE(DeepSeekBoundaryCreditTracker::Create(3).ok());
}

}  // namespace
}  // namespace pih
