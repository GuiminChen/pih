#include "pih/model/qwen3_kv_admission_transaction.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

QwenKvSlotPool clean_pool(std::uint32_t slots) {
  auto pool = QwenKvSlotPool::Create(
      slots, static_cast<std::uint64_t>(slots) *
                 QwenKvSlotPool::kSlotPayloadBytes,
      static_cast<std::uint64_t>(slots) * sizeof(QwenKvSlotState));
  EXPECT_TRUE(pool.ok());
  EXPECT_TRUE(pool->complete_startup_sanitize(
      static_cast<std::uint64_t>(slots) * QwenKvSlotPool::kSlotPayloadBytes,
      static_cast<std::uint64_t>(slots) * sizeof(QwenKvSlotState), true).ok());
  return std::move(*pool);
}

TEST(QwenKvAdmissionTransactionTest, PublishesExactLifetimeReservation) {
  auto pool = clean_pool(3);
  auto admission = QwenKvAdmissionTransaction::Reserve(pool, 7, 17);
  ASSERT_TRUE(admission.ok()) << admission.status().message();
  EXPECT_EQ(admission->state(), QwenKvAdmissionState::kReserved);
  EXPECT_EQ(pool.clean_credits(), 1U);
  EXPECT_EQ(pool.lifecycle_count(QwenKvSlotLifecycle::kAdmissionReserved), 2U);
  EXPECT_EQ(admission->block_table(), nullptr);

  ASSERT_TRUE(admission->publish().ok());
  ASSERT_NE(admission->block_table(), nullptr);
  EXPECT_EQ(admission->state(), QwenKvAdmissionState::kPublished);
  EXPECT_EQ(admission->block_table()->descriptor().reserved_tokens, 17U);
  EXPECT_EQ(admission->block_table()->descriptor().sequence_generation,
            admission->sequence_generation());
  EXPECT_EQ(pool.lifecycle_count(QwenKvSlotLifecycle::kOwned), 2U);
  EXPECT_EQ(admission->rollback().code(), StatusCode::kFailedPrecondition);
}

TEST(QwenKvAdmissionTransactionTest, RollbackIsIdempotentAndKeepsGeneration) {
  auto pool = clean_pool(2);
  auto first = QwenKvAdmissionTransaction::Reserve(pool, 3, 16);
  ASSERT_TRUE(first.ok());
  const auto first_generation = first->sequence_generation();
  const auto first_handle_generation = pool.slot(0).generation;
  ASSERT_TRUE(first->rollback().ok());
  ASSERT_TRUE(first->rollback().ok());
  EXPECT_EQ(pool.clean_credits(), 2U);
  EXPECT_EQ(pool.lifecycle_count(QwenKvSlotLifecycle::kFreeClean), 2U);

  auto second = QwenKvAdmissionTransaction::Reserve(pool, 3, 16);
  ASSERT_TRUE(second.ok());
  EXPECT_GT(second->sequence_generation(), first_generation);
  ASSERT_TRUE(second->publish().ok());
  EXPECT_GT(second->block_table()->reserved_handles()[0].generation,
            first_handle_generation);
}

TEST(QwenKvAdmissionTransactionTest, RejectsBoundsAndCleanCreditShortage) {
  auto pool = clean_pool(1);
  EXPECT_FALSE(QwenKvAdmissionTransaction::Reserve(pool, 0, 17).ok());
  EXPECT_FALSE(QwenKvAdmissionTransaction::Reserve(
      pool, 0, QwenKvBlockTable::kMaximumReservedTokens + 1).ok());
  EXPECT_FALSE(QwenKvAdmissionTransaction::Reserve(
      pool, QwenKvSlotPool::kNoOwner, 1).ok());
  EXPECT_EQ(pool.clean_credits(), 1U);
}

}  // namespace
}  // namespace pih
