#include "pih/model/qwen3_kv_slot_pool.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

QwenKvSlotPool pool(std::uint32_t slots = 4) {
  return QwenKvSlotPool::Create(
             slots, slots * QwenKvSlotPool::kSlotPayloadBytes,
             slots * sizeof(QwenKvSlotState))
      .value();
}

TEST(QwenKvSlotPoolTest, FreezesWireSizesAndRejectsInvalidBacking) {
  EXPECT_EQ(sizeof(QwenKvBlockHandle), 8);
  EXPECT_EQ(sizeof(QwenKvSlotState), 16);
  EXPECT_FALSE(QwenKvSlotPool::Create(0, 0, 0).ok());
  EXPECT_FALSE(QwenKvSlotPool::Create(
                   2, QwenKvSlotPool::kSlotPayloadBytes,
                   2 * sizeof(QwenKvSlotState))
                   .ok());
}

TEST(QwenKvSlotPoolTest, PublishesNoCreditsBeforeFullStartupBarrier) {
  auto subject = pool();
  EXPECT_FALSE(subject.ready());
  EXPECT_EQ(subject.clean_credits(), 0);
  EXPECT_FALSE(subject.reserve(7, 1).ok());
  EXPECT_FALSE(subject.complete_startup_sanitize(
                          4 * QwenKvSlotPool::kSlotPayloadBytes - 1,
                          4 * sizeof(QwenKvSlotState), true)
                   .ok());
  EXPECT_TRUE(subject.failed());
  EXPECT_EQ(subject.clean_credits(), 0);
}

TEST(QwenKvSlotPoolTest, ReservesPublishesAndRollsBackAtomically) {
  auto subject = pool();
  ASSERT_TRUE(subject.complete_startup_sanitize(
                         4 * QwenKvSlotPool::kSlotPayloadBytes,
                         4 * sizeof(QwenKvSlotState), true)
                  .ok());
  auto reserved = subject.reserve(9, 2);
  ASSERT_TRUE(reserved.ok());
  EXPECT_EQ(subject.clean_credits(), 2);
  EXPECT_EQ(subject.slot(0).generation, 2);
  EXPECT_EQ(subject.slot(0).state,
            QwenKvSlotLifecycle::kAdmissionReserved);
  EXPECT_FALSE(subject.publish(9, {reserved->data(), 1}).ok());
  EXPECT_FALSE(subject.reserve(9, 1).ok());
  EXPECT_EQ(subject.slot(0).state,
            QwenKvSlotLifecycle::kAdmissionReserved);
  ASSERT_TRUE(subject.publish(9, reserved.value()).ok());
  EXPECT_EQ(subject.slot(0).state, QwenKvSlotLifecycle::kOwned);

  auto rollback = subject.reserve(10, 2);
  ASSERT_TRUE(rollback.ok());
  const auto retained_generation = rollback->front().generation;
  ASSERT_TRUE(subject.rollback(10, rollback.value()).ok());
  EXPECT_EQ(subject.clean_credits(), 2);
  EXPECT_EQ(subject.slot(2).generation, retained_generation);
  EXPECT_EQ(subject.slot(2).state, QwenKvSlotLifecycle::kFreeClean);
}

TEST(QwenKvSlotPoolTest, ExhaustionAndStaleHandleDoNotPartiallyMutate) {
  auto subject = pool(2);
  ASSERT_TRUE(subject.complete_startup_sanitize(
                         2 * QwenKvSlotPool::kSlotPayloadBytes,
                         2 * sizeof(QwenKvSlotState), true)
                  .ok());
  auto first = subject.reserve(1, 1);
  ASSERT_TRUE(first.ok());
  EXPECT_FALSE(subject.reserve(2, 2).ok());
  EXPECT_EQ(subject.clean_credits(), 1);
  auto stale = first.value();
  stale[0].generation--;
  EXPECT_FALSE(subject.publish(1, stale).ok());
  EXPECT_EQ(subject.slot(0).state,
            QwenKvSlotLifecycle::kAdmissionReserved);
  EXPECT_EQ(subject.clean_credits(), 1);
}

TEST(QwenKvSlotPoolTest, ReclaimsOnlyAfterMatchingLastUseEventAndFullScrub) {
  auto subject = pool(2);
  ASSERT_TRUE(subject.complete_startup_sanitize(
                         2 * QwenKvSlotPool::kSlotPayloadBytes,
                         2 * sizeof(QwenKvSlotState), true)
                  .ok());
  auto handles = subject.reserve(3, 1);
  ASSERT_TRUE(handles.ok());
  ASSERT_TRUE(subject.publish(3, handles.value()).ok());
  ASSERT_TRUE(subject.commit_valid_tokens(3, handles->front(), 16).ok());
  const QwenKvCompletionEvent last_use{91, 4};
  ASSERT_TRUE(subject.release(3, handles.value(), last_use).ok());
  EXPECT_EQ(subject.clean_credits(), 1);
  EXPECT_FALSE(subject.complete_reclaim_event(
                          handles->front(), {91, 3}, true, true)
                   .ok());
  EXPECT_EQ(subject.slot(0).state, QwenKvSlotLifecycle::kReclaimPending);
  EXPECT_EQ(subject.complete_reclaim_event(
                handles->front(), last_use, false, true).code(),
            StatusCode::kUnavailable);
  ASSERT_TRUE(subject.complete_reclaim_event(
                         handles->front(), last_use, true, true)
                  .ok());
  EXPECT_EQ(subject.slot(0).state, QwenKvSlotLifecycle::kFreeDirty);
  EXPECT_EQ(subject.clean_credits(), 1);

  const QwenKvCompletionEvent scrub_event{92, 8};
  auto work = subject.begin_next_scrub(scrub_event);
  ASSERT_TRUE(work.ok());
  EXPECT_EQ(work->slot, handles->front());
  EXPECT_EQ(work->bytes, QwenKvSlotPool::kSlotPayloadBytes);
  EXPECT_FALSE(subject.begin_next_scrub({93, 1}).ok());
  EXPECT_EQ(subject.complete_scrub(work->slot, scrub_event,
                                   QwenKvSlotPool::kSlotPayloadBytes, false,
                                   true)
                .code(),
            StatusCode::kUnavailable);
  ASSERT_TRUE(subject.complete_scrub(
                         work->slot, scrub_event,
                         QwenKvSlotPool::kSlotPayloadBytes, true, true)
                  .ok());
  EXPECT_EQ(subject.slot(0).state, QwenKvSlotLifecycle::kFreeClean);
  EXPECT_EQ(subject.clean_credits(), 2);
}

TEST(QwenKvSlotPoolTest, ProjectsPrefixWithoutPublishingIt) {
  auto subject = pool(3);
  ASSERT_TRUE(subject.complete_startup_sanitize(
      3 * QwenKvSlotPool::kSlotPayloadBytes,
      3 * sizeof(QwenKvSlotState), true).ok());
  auto handles = subject.reserve(7, 2).value();
  ASSERT_TRUE(subject.publish(7, handles).ok());

  auto projected = subject.project_prefix(7, handles, 18);

  ASSERT_TRUE(projected.ok()) << projected.status().message();
  EXPECT_EQ((*projected)[handles[0].slot].valid_tokens, 16);
  EXPECT_EQ((*projected)[handles[1].slot].valid_tokens, 2);
  EXPECT_EQ(subject.slot(handles[0].slot).valid_tokens, 0);
  EXPECT_EQ(subject.slot(handles[1].slot).valid_tokens, 0);
  auto stale = handles;
  --stale[1].generation;
  EXPECT_FALSE(subject.project_prefix(7, stale, 18).ok());
}

TEST(QwenKvSlotPoolTest, ScrubFailureFailStopsWithoutPublishingCredit) {
  auto subject = pool(1);
  ASSERT_TRUE(subject.complete_startup_sanitize(
                         QwenKvSlotPool::kSlotPayloadBytes,
                         sizeof(QwenKvSlotState), true)
                  .ok());
  auto handles = subject.reserve(5, 1);
  ASSERT_TRUE(handles.ok());
  ASSERT_TRUE(subject.publish(5, handles.value()).ok());
  ASSERT_TRUE(subject.release(5, handles.value(), {10, 1}).ok());
  ASSERT_TRUE(subject.complete_reclaim_event(
                         handles->front(), {10, 1}, true, true)
                  .ok());
  auto work = subject.begin_next_scrub({11, 1});
  ASSERT_TRUE(work.ok());
  EXPECT_FALSE(subject.complete_scrub(
                          work->slot, {11, 1},
                          QwenKvSlotPool::kSlotPayloadBytes - 1, true, true)
                   .ok());
  EXPECT_TRUE(subject.failed());
  EXPECT_EQ(subject.clean_credits(), 0);
  EXPECT_EQ(subject.slot(0).state, QwenKvSlotLifecycle::kScrubbing);
  EXPECT_FALSE(subject.reserve(6, 1).ok());
}

}  // namespace
}  // namespace pih
