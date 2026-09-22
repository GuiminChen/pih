#include "pih/model/qwen3_kv_sequence_lease.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace pih {
namespace {

QwenKvSlotPool ready_pool(std::uint32_t slots) {
  auto pool = QwenKvSlotPool::Create(
                  slots, slots * QwenKvSlotPool::kSlotPayloadBytes,
                  slots * sizeof(QwenKvSlotState))
                  .value();
  EXPECT_TRUE(pool.complete_startup_sanitize(
                      slots * QwenKvSlotPool::kSlotPayloadBytes,
                      slots * sizeof(QwenKvSlotState), true)
                  .ok());
  return pool;
}

TEST(QwenKvSequenceLeaseTest, AdmitsAndCommitsPhysicalAndLogicalPrefixTogether) {
  auto pool = ready_pool(4);
  auto lease = QwenKvSequenceLease::Admit(pool, 7, 3, 33);
  ASSERT_TRUE(lease.ok()) << lease.status().message();
  EXPECT_EQ(pool.clean_credits(), 1);
  auto plan = lease->prepare_append(17);
  ASSERT_TRUE(plan.ok());
  ASSERT_TRUE(lease->commit_append(plan.value()).ok());
  EXPECT_EQ(lease->block_table().visible_handles().size(), 2);
  const auto handles = lease->block_table().reserved_handles();
  EXPECT_EQ(pool.slot(handles[0].slot).valid_tokens, 16);
  EXPECT_EQ(pool.slot(handles[1].slot).valid_tokens, 1);
  EXPECT_EQ(pool.slot(handles[2].slot).valid_tokens, 0);
}

TEST(QwenKvSequenceLeaseTest, RollbackAndStalePlanPreserveBothLedgers) {
  auto pool = ready_pool(3);
  auto lease = QwenKvSequenceLease::Admit(pool, 4, 2, 33).value();
  auto plan = lease.prepare_append(16).value();
  ASSERT_TRUE(lease.rollback_append(plan).ok());
  EXPECT_EQ(lease.block_table().descriptor().committed_tokens, 0);
  for (const auto& handle : lease.block_table().reserved_handles()) {
    EXPECT_EQ(pool.slot(handle.slot).valid_tokens, 0);
  }
  ASSERT_TRUE(lease.commit_append(plan).ok());
  EXPECT_FALSE(lease.commit_append(plan).ok());
  EXPECT_EQ(lease.block_table().descriptor().committed_tokens, 16);
}

TEST(QwenKvSequenceLeaseTest, ReleaseMakesEveryReservedSlotUnadmittable) {
  auto pool = ready_pool(3);
  auto lease = QwenKvSequenceLease::Admit(pool, 4, 2, 33).value();
  ASSERT_TRUE(lease.release({77, 1}).ok());
  EXPECT_TRUE(lease.released());
  EXPECT_EQ(pool.clean_credits(), 0);
  EXPECT_EQ(pool.lifecycle_count(QwenKvSlotLifecycle::kReclaimPending), 3);
  EXPECT_FALSE(lease.prepare_append(1).ok());
  EXPECT_FALSE(lease.release({78, 1}).ok());
}

}  // namespace
}  // namespace pih
