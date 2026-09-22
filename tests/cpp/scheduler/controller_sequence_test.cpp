#include "pih/scheduler/controller_sequence.h"

#include <gtest/gtest.h>

namespace pih {

TEST(ControllerSequenceTest, CommitsChunkedPrefillThenDecodeByEvents) {
  auto sequence = ControllerSequence::Admit(7, 5, 10).value();
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kQueuedPrefill);
  EXPECT_EQ(sequence.ready_phase().value(), PackedTokenPhase::kPrefill);
  EXPECT_EQ(sequence.next_real_token_count(3, 0).value(), 3U);
  EXPECT_FALSE(sequence.next_plan_produces_logits(3, 0).value());
  EXPECT_TRUE(sequence.prepare(1, PackedTokenPhase::kPrefill, 3).ok());
  EXPECT_TRUE(sequence.commit(1).ok());
  EXPECT_TRUE(sequence.mark_in_flight(1).ok());
  EXPECT_TRUE(sequence.complete({1, 7, 3, 0, 11, PackedTokenPhase::kDecode, false}).ok());
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kQueuedPrefill);
  EXPECT_EQ(sequence.next_real_token_count(3, 0).value(), 2U);
  EXPECT_TRUE(sequence.next_plan_produces_logits(3, 0).value());
  EXPECT_TRUE(sequence.prepare(2, PackedTokenPhase::kPrefill, 2).ok());
  EXPECT_TRUE(sequence.commit(2).ok());
  EXPECT_TRUE(sequence.mark_in_flight(2).ok());
  EXPECT_TRUE(sequence.complete({2, 7, 5, 1, 12, PackedTokenPhase::kDecode, false}).ok());
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kReadyDecode);
  EXPECT_EQ(sequence.ready_phase().value(), PackedTokenPhase::kDecode);
  EXPECT_EQ(sequence.next_real_token_count(3, 0).value(), 1U);
  EXPECT_TRUE(sequence.prepare(3, PackedTokenPhase::kDecode, 1).ok());
  EXPECT_TRUE(sequence.commit(3).ok());
  EXPECT_TRUE(sequence.mark_in_flight(3).ok());
  EXPECT_TRUE(sequence.complete({3, 7, 6, 2, 13, PackedTokenPhase::kVerify, false}).ok());
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kReadyDecode);
  EXPECT_EQ(sequence.ready_phase().value(), PackedTokenPhase::kVerify);
  EXPECT_EQ(sequence.next_real_token_count(3, 5).value(), 5U);
  EXPECT_TRUE(sequence.next_plan_produces_logits(3, 5).value());
  EXPECT_TRUE(sequence.prepare(4, PackedTokenPhase::kVerify, 5).ok());
}

TEST(ControllerSequenceTest, PreparedAbortDoesNotAdvanceCommittedState) {
  auto sequence = ControllerSequence::Admit(1, 4, 1).value();
  EXPECT_TRUE(sequence.prepare(4, PackedTokenPhase::kPrefill, 2).ok());
  EXPECT_TRUE(sequence.abort_prepared(4).ok());
  EXPECT_EQ(sequence.model_processed_length(), 0U);
  EXPECT_EQ(sequence.state_generation(), 1U);
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kQueuedPrefill);
}

TEST(ControllerSequenceTest, CancelDrainsCommittedWorkBeforeTerminal) {
  auto sequence = ControllerSequence::Admit(1, 1, 1).value();
  ASSERT_TRUE(sequence.prepare(1, PackedTokenPhase::kPrefill, 1).ok());
  ASSERT_TRUE(sequence.commit(1).ok());
  ASSERT_TRUE(sequence.mark_in_flight(1).ok());
  EXPECT_TRUE(sequence.request_cancel().ok());
  EXPECT_FALSE(sequence.begin_draining().ok());
  EXPECT_TRUE(sequence.complete({1, 1, 1, 1, 2, PackedTokenPhase::kDecode, false}).ok());
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kDraining);
  EXPECT_TRUE(sequence.finish_cancel().ok());
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kCancelled);
}

TEST(ControllerSequenceTest, RejectsStaleEventsAndLedgerCuts) {
  auto sequence = ControllerSequence::Admit(1, 2, 1).value();
  ASSERT_TRUE(sequence.prepare(2, PackedTokenPhase::kPrefill, 2).ok());
  EXPECT_FALSE(sequence.commit(1).ok());
  ASSERT_TRUE(sequence.commit(2).ok());
  EXPECT_FALSE(sequence.complete({2, 1, 2, 1, 2, PackedTokenPhase::kDecode, false}).ok());
  ASSERT_TRUE(sequence.mark_in_flight(2).ok());
  EXPECT_FALSE(sequence.complete({1, 1, 2, 1, 2, PackedTokenPhase::kDecode, false}).ok());
  EXPECT_FALSE(sequence.complete({2, 1, 2, 2, 2, PackedTokenPhase::kDecode, false}).ok());
  EXPECT_EQ(sequence.model_processed_length(), 0U);
  EXPECT_EQ(sequence.state_generation(), 1U);
}

TEST(ControllerSequenceTest, CancelledPreparedPlanCannotPublishCompletion) {
  auto sequence = ControllerSequence::Admit(1, 1, 1).value();
  ASSERT_TRUE(sequence.prepare(1, PackedTokenPhase::kPrefill, 1).ok());
  ASSERT_TRUE(sequence.request_cancel().ok());
  EXPECT_FALSE(sequence.complete(
      {1, 1, 1, 1, 2, PackedTokenPhase::kDecode, false}).ok());
  EXPECT_EQ(sequence.model_processed_length(), 0U);
  EXPECT_TRUE(sequence.begin_draining().ok());
}

TEST(ControllerSequenceTest, PreparedCancelCanRollbackAndFailIsTerminal) {
  auto sequence = ControllerSequence::Admit(1, 2, 1).value();
  ASSERT_TRUE(sequence.prepare(1, PackedTokenPhase::kPrefill, 1).ok());
  ASSERT_TRUE(sequence.request_cancel().ok());
  EXPECT_TRUE(sequence.begin_draining().ok());
  EXPECT_TRUE(sequence.finish_cancel().ok());
  EXPECT_FALSE(sequence.fail().ok());
}

TEST(ControllerSequenceTest, TerminalDrainCannotBeMisclassifiedAsCancel) {
  auto sequence = ControllerSequence::Admit(1, 1, 1).value();
  ASSERT_TRUE(sequence.prepare(1, PackedTokenPhase::kPrefill, 1).ok());
  ASSERT_TRUE(sequence.commit(1).ok());
  ASSERT_TRUE(sequence.mark_in_flight(1).ok());
  ASSERT_TRUE(sequence.complete(
      {1, 1, 1, 1, 2, PackedTokenPhase::kDecode, true}).ok());
  EXPECT_FALSE(sequence.finish_cancel().ok());
  EXPECT_TRUE(sequence.finish_drain().ok());
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kCompleted);
}

TEST(ControllerSequenceTest, SchedulingDerivationRejectsNonReadyAndAmbiguousVerify) {
  auto sequence = ControllerSequence::Admit(1, 1, 1).value();
  EXPECT_FALSE(sequence.next_real_token_count(0, 0).ok());
  ASSERT_TRUE(sequence.prepare(1, PackedTokenPhase::kPrefill, 1).ok());
  EXPECT_FALSE(sequence.ready_phase().ok());
  EXPECT_FALSE(sequence.next_plan_produces_logits(1, 0).ok());
}

}  // namespace pih
