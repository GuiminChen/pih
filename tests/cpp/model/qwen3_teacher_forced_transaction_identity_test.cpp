#include <cstdint>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_transaction_identity.h"

namespace pih {

TEST(QwenTeacherForcedTransactionIdentityTest,
     ReservesThreeCopiesFrontierAndEventAtomically) {
  auto value = reserve_qwen_teacher_forced_transaction_identity(100, 200).value();
  EXPECT_EQ(value.row_upload_plan_id, 100U);
  EXPECT_EQ(value.target_upload_plan_id, 101U);
  EXPECT_EQ(value.readback_plan_id, 102U);
  EXPECT_EQ(value.frontier_plan_id, 103U);
  EXPECT_EQ(value.next_plan_id, 104U);
  EXPECT_EQ(value.event_generation, 200U);
  EXPECT_EQ(value.next_event_generation, 201U);
}

TEST(QwenTeacherForcedTransactionIdentityTest,
     BuildsMatchingIdleSlotAndFrontier) {
  const auto identity =
      reserve_qwen_teacher_forced_transaction_identity(100, 200).value();
  auto completion = make_qwen_teacher_forced_transaction_completion(
      identity, 31, 17, 9, 0, 77, 1'000, 2'000);
  ASSERT_TRUE(completion.ok()) << completion.status().message();
  EXPECT_EQ(completion->slot.state(), CompletionEventSlotState::kIdle);
  EXPECT_EQ(completion->slot.context_identity(), 17U);
  EXPECT_EQ(completion->frontier.event_generation(), 200U);
  EXPECT_EQ(completion->frontier.key().plan_generation, 103U);
}

TEST(QwenTeacherForcedTransactionIdentityTest, RejectsEveryOverflowEdge) {
  EXPECT_FALSE(reserve_qwen_teacher_forced_transaction_identity(0, 1).ok());
  EXPECT_FALSE(reserve_qwen_teacher_forced_transaction_identity(1, 0).ok());
  EXPECT_FALSE(reserve_qwen_teacher_forced_transaction_identity(
                   UINT64_MAX - 3, 1).ok());
  EXPECT_FALSE(reserve_qwen_teacher_forced_transaction_identity(
                   1, UINT64_MAX).ok());
}

}  // namespace pih
