#include "pih/model/deepseek_attention_sequence_transaction.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class Operations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return result;
  }
  DeepSeekExpertAsyncStatus result = DeepSeekExpertAsyncStatus::kSuccess;
};

struct Fixture final {
  Operations operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 1024}, {0x2000, 1024}, 5, operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(4).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(4).value();
};

TEST(DeepSeekAttentionSequenceTransactionTest, CommitsAllOwnersTogether) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 2, 2, fixture.banks, fixture.ratio4, fixture.ratio128);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->begin(9).ok());
  ASSERT_TRUE(transaction->reserve_ratio4_append(2, 0).ok());
  ASSERT_TRUE(transaction->reserve_ratio128_append(3, 0).ok());
  ASSERT_TRUE(transaction->seal(9).ok());
  ASSERT_TRUE(transaction->poll().ok());
  EXPECT_TRUE(transaction->commit().ok());
  EXPECT_EQ(fixture.banks.generation(), 2U);
  EXPECT_EQ(fixture.ratio4.published_pairs(), 1U);
  EXPECT_EQ(fixture.ratio128.published_pages(), 1U);
}

TEST(DeepSeekAttentionSequenceTransactionTest,
     ExternalKernelErrorPreventsGenerationPublish) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 1, 1, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  std::uint32_t kernel_error = 0;
  ASSERT_TRUE(transaction.begin(9).ok());
  auto first_claim = transaction.claim_external_error_channel(&kernel_error, 100);
  ASSERT_TRUE(first_claim.ok());
  EXPECT_TRUE(*first_claim);
  auto repeated_claim = transaction.claim_external_error_channel(&kernel_error, 100);
  ASSERT_TRUE(repeated_claim.ok());
  EXPECT_FALSE(*repeated_claim);
  std::uint32_t other_error = 0;
  auto second_claim =
      transaction.claim_external_error_channel(&other_error, 100);
  ASSERT_TRUE(second_claim.ok());
  EXPECT_TRUE(*second_claim);
  EXPECT_FALSE(transaction.claim_external_error_channel(&kernel_error, 101).ok());
  ASSERT_TRUE(transaction.reserve_ratio4_append(2, 0).ok());
  ASSERT_TRUE(transaction.reserve_ratio128_append(3, 0).ok());
  ASSERT_TRUE(transaction.seal(9).ok());
  other_error = 4;
  EXPECT_EQ(transaction.poll().value(), DeepSeekExpertAsyncStatus::kError);
  EXPECT_EQ(transaction.state(),
            DeepSeekAttentionSequenceTransactionState::kPoisoned);
  EXPECT_EQ(fixture.ratio4.published_pairs(), 0U);
  EXPECT_EQ(fixture.ratio128.published_pages(), 0U);
  EXPECT_FALSE(transaction.commit().ok());
}

TEST(DeepSeekAttentionSequenceTransactionTest, AbortRollsBackEveryOwner) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 2, 2, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(9).ok());
  ASSERT_TRUE(transaction.reserve_ratio4_append(2, 0).ok());
  ASSERT_TRUE(transaction.reserve_ratio128_append(3, 0).ok());
  ASSERT_TRUE(transaction.seal(9).ok());
  ASSERT_TRUE(transaction.poll().ok());
  EXPECT_TRUE(transaction.abort().ok());
  EXPECT_EQ(fixture.banks.generation(), 1U);
  EXPECT_EQ(fixture.ratio4.free_pairs(), 4U);
  EXPECT_EQ(fixture.ratio128.free_pages(), 4U);
}

TEST(DeepSeekAttentionSequenceTransactionTest, WaitsBeforeAnyPublication) {
  Fixture fixture;
  fixture.operations.result = DeepSeekExpertAsyncStatus::kInProgress;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 1, 1, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(9).ok());
  ASSERT_TRUE(transaction.reserve_ratio4_append(2, 0).ok());
  ASSERT_TRUE(transaction.seal(9).ok());
  EXPECT_EQ(transaction.poll().value(), DeepSeekExpertAsyncStatus::kInProgress);
  EXPECT_FALSE(transaction.commit().ok());
  EXPECT_EQ(fixture.ratio4.published_pairs(), 0U);
  EXPECT_EQ(fixture.ratio4.reserved_pairs(), 1U);
}

TEST(DeepSeekAttentionSequenceTransactionTest, SealRequiresBeginStream) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 0, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  EXPECT_FALSE(transaction.begin(0).ok());
  ASSERT_TRUE(transaction.begin(9).ok());
  EXPECT_EQ(transaction.stream(), 9U);
  EXPECT_FALSE(transaction.seal(10).ok());
  EXPECT_TRUE(transaction.seal(9).ok());
}

TEST(DeepSeekAttentionSequenceTransactionTest, EnforcesPreallocatedBounds) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 1, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(9).ok());
  ASSERT_TRUE(transaction.reserve_ratio4_append(2, 0).ok());
  EXPECT_FALSE(transaction.reserve_ratio4_append(2, 1).ok());
  EXPECT_FALSE(transaction.reserve_ratio128_append(3, 0).ok());
}

TEST(DeepSeekAttentionSequenceTransactionTest,
     PreflightFailurePublishesNoEarlierOwner) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 1, 1, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(9).ok());
  auto ratio4 = transaction.reserve_ratio4_append(2, 0);
  auto ratio128 = transaction.reserve_ratio128_append(3, 0);
  ASSERT_TRUE(ratio4.ok() && ratio128.ok());
  ASSERT_TRUE(fixture.ratio128.rollback(*ratio128).ok());
  ASSERT_TRUE(transaction.seal(9).ok());
  ASSERT_TRUE(transaction.poll().ok());
  EXPECT_FALSE(transaction.commit().ok());
  EXPECT_EQ(fixture.ratio4.published_pairs(), 0U);
  EXPECT_EQ(fixture.ratio4.reserved_pairs(), 1U);
  EXPECT_EQ(fixture.banks.generation(), 1U);
}

TEST(DeepSeekAttentionSequenceTransactionTest,
     CommitsTailCowWithFixedGeneration) {
  Fixture fixture;
  auto old4 = fixture.ratio4.reserve(7, 2, 3);
  auto old128 = fixture.ratio128.reserve(7, 3, 5);
  ASSERT_TRUE(old4.ok() && old128.ok());
  ASSERT_TRUE(fixture.ratio4.publish(*old4).ok());
  ASSERT_TRUE(fixture.ratio128.publish(*old128).ok());
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 1, 1, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(9).ok());
  auto next4 = transaction.reserve_ratio4_tail_cow(*old4);
  auto next128 = transaction.reserve_ratio128_tail_cow(*old128);
  ASSERT_TRUE(next4.ok() && next128.ok());
  ASSERT_TRUE(transaction.seal(9).ok());
  ASSERT_TRUE(transaction.poll().ok());
  ASSERT_TRUE(transaction.commit().ok());
  EXPECT_EQ(fixture.banks.generation(), 2U);
  EXPECT_FALSE(fixture.ratio4.release(*old4).ok());
  EXPECT_FALSE(fixture.ratio128.release(*old128).ok());
  EXPECT_TRUE(fixture.ratio4.release(*next4).ok());
  EXPECT_TRUE(fixture.ratio128.release(*next128).ok());
}

TEST(DeepSeekAttentionSequenceTransactionTest,
     MaterializesTentativePagedIndexerViewWithoutLogicalHoles) {
  Fixture fixture;
  auto committed = fixture.ratio4.reserve(7, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed).ok());
  auto occupied = fixture.ratio4.reserve(8, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(occupied).ok());
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 1, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(9).ok());
  auto appended = transaction.reserve_ratio4_append(2, 1).value();
  auto slots = transaction.ratio4_index_page_slots(2, 2);
  ASSERT_TRUE(slots.ok()) << slots.status().message();
  EXPECT_EQ(*slots, std::vector<std::uint32_t>(
                        {committed.index.slot(), appended.index.slot()}));
  auto main_slots = transaction.ratio4_main_page_slots(2, 2);
  ASSERT_TRUE(main_slots.ok()) << main_slots.status().message();
  EXPECT_EQ(*main_slots, std::vector<std::uint32_t>(
                             {committed.main.slot(), appended.main.slot()}));
  EXPECT_EQ(transaction.ratio4_physical_page_count(), 4U);
  EXPECT_FALSE(transaction.ratio4_index_page_slots(2, 3).ok());
  EXPECT_FALSE(transaction.ratio4_main_page_slots(2, 3).ok());
}

TEST(DeepSeekAttentionSequenceTransactionTest,
     MaterializesTentativeRatio128MainPagesWithoutLogicalHoles) {
  Fixture fixture;
  auto committed = fixture.ratio128.reserve(7, 3, 0).value();
  ASSERT_TRUE(fixture.ratio128.publish(committed).ok());
  auto occupied = fixture.ratio128.reserve(8, 3, 0).value();
  ASSERT_TRUE(fixture.ratio128.publish(occupied).ok());
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 0, 1, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(9).ok());
  auto appended = transaction.reserve_ratio128_append(3, 1).value();
  auto slots = transaction.ratio128_main_page_slots(3, 2);
  ASSERT_TRUE(slots.ok()) << slots.status().message();
  EXPECT_EQ(*slots, std::vector<std::uint32_t>(
                        {committed.handle.slot(), appended.handle.slot()}));
  EXPECT_EQ(transaction.ratio128_physical_page_count(), 4U);
  EXPECT_FALSE(transaction.ratio128_main_page_slots(3, 3).ok());
}

} }  // namespace pih
