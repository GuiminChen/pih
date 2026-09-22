#include "pih/model/deepseek_attention_plan_compute_driver.h"

#include <gtest/gtest.h>

#include <array>

namespace pih { namespace {

class FixedOps final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override {
    ++copies;
    return fail_copy ? Status::Internal("injected fixed-state copy failure")
                     : Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    ++records; return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    ++queries; return result;
  }
  DeepSeekExpertAsyncStatus result = DeepSeekExpertAsyncStatus::kSuccess;
  std::uint32_t copies = 0;
  std::uint32_t records = 0;
  std::uint32_t queries = 0;
  bool fail_copy = false;
};

class Inner final : public DeepSeekStageComputeDriver {
 public:
  Status launch(const DeepSeekPipelinePlanDescriptor&,
                const DeepSeekStagePlan&) override {
    ++launches;
    if (reserve_pages) {
      first = first_transaction->reserve_ratio4_append(2, 0).value();
      second = second_transaction->reserve_ratio4_append(2, 0).value();
    }
    return fail_launch ? Status::Internal("injected inner launch failure")
                       : Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override { return status; }
  DeepSeekAttentionSequenceTransaction* first_transaction = nullptr;
  DeepSeekAttentionSequenceTransaction* second_transaction = nullptr;
  DeepSeekRatio4PagePair first;
  DeepSeekRatio4PagePair second;
  DeepSeekStageComputeStatus status = DeepSeekStageComputeStatus::kSuccess;
  bool reserve_pages = false;
  bool fail_launch = false;
  std::uint32_t launches = 0;
};

struct Fixture final {
  FixedOps first_ops;
  FixedOps second_ops;
  DeepSeekFixedStateBanks first_banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 1024}, {0x2000, 1024}, 11, first_ops).value();
  DeepSeekFixedStateBanks second_banks = DeepSeekFixedStateBanks::Create(
      {0x3000, 1024}, {0x4000, 1024}, 12, second_ops).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(4).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(4).value();
  DeepSeekAttentionSequenceTransaction first =
      DeepSeekAttentionSequenceTransaction::Create(
          1, 1, 0, first_banks, ratio4, ratio128).value();
  DeepSeekAttentionSequenceTransaction second =
      DeepSeekAttentionSequenceTransaction::Create(
          2, 1, 0, second_banks, ratio4, ratio128).value();
  Inner inner;
  std::array<DeepSeekAttentionSequenceTransaction*, 2> transactions{
      &first, &second};
  DeepSeekAttentionPlanComputeDriver driver =
      DeepSeekAttentionPlanComputeDriver::Create(inner, transactions, 9)
          .value();

  Fixture() {
    inner.first_transaction = &first;
    inner.second_transaction = &second;
  }
};

DeepSeekPipelinePlanDescriptor plan() {
  return {1, 1, DeepSeekPlanPhase::kDecode, 2, 2};
}

DeepSeekStagePlan stage() {
  return {0, {0, 42}, true, true, false};
}

TEST(DeepSeekAttentionPlanComputeDriverTest,
     CommitsEverySequenceOnlyAfterAllCompletionEvidence) {
  Fixture fixture;
  fixture.first_ops.result = DeepSeekExpertAsyncStatus::kInProgress;
  fixture.second_ops.result = DeepSeekExpertAsyncStatus::kInProgress;
  ASSERT_TRUE(fixture.driver.launch(plan(), stage()).ok());
  EXPECT_EQ(fixture.first.state(),
            DeepSeekAttentionSequenceTransactionState::kPreparing);
  EXPECT_EQ(fixture.driver.poll().value(),
            DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(fixture.first_banks.generation(), 1U);
  EXPECT_EQ(fixture.second_banks.generation(), 1U);
  fixture.first_ops.result = DeepSeekExpertAsyncStatus::kSuccess;
  fixture.second_ops.result = DeepSeekExpertAsyncStatus::kSuccess;
  EXPECT_EQ(fixture.driver.poll().value(),
            DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(fixture.first_banks.generation(), 2U);
  EXPECT_EQ(fixture.second_banks.generation(), 2U);
}

TEST(DeepSeekAttentionPlanComputeDriverTest,
     InnerFailureDrainsThenAbortsAllTentativeState) {
  Fixture fixture;
  fixture.inner.status = DeepSeekStageComputeStatus::kError;
  ASSERT_TRUE(fixture.driver.launch(plan(), stage()).ok());
  EXPECT_EQ(fixture.driver.poll().value(), DeepSeekStageComputeStatus::kError);
  EXPECT_EQ(fixture.first_banks.generation(), 1U);
  EXPECT_EQ(fixture.second_banks.generation(), 1U);
  EXPECT_EQ(fixture.first.state(),
            DeepSeekAttentionSequenceTransactionState::kIdle);
  EXPECT_EQ(fixture.second.state(),
            DeepSeekAttentionSequenceTransactionState::kIdle);
}

TEST(DeepSeekAttentionPlanComputeDriverTest,
     PreflightsEveryCommitBeforePublishingTheFirstSequence) {
  Fixture fixture;
  fixture.inner.reserve_pages = true;
  ASSERT_TRUE(fixture.driver.launch(plan(), stage()).ok());
  ASSERT_TRUE(fixture.ratio4.rollback(fixture.inner.second).ok());
  EXPECT_EQ(fixture.driver.poll().value(), DeepSeekStageComputeStatus::kError);
  EXPECT_EQ(fixture.ratio4.published_pairs(), 0U);
  EXPECT_EQ(fixture.first_banks.generation(), 1U);
  EXPECT_EQ(fixture.second_banks.generation(), 1U);
}

TEST(DeepSeekAttentionPlanComputeDriverTest,
     RejectsTransactionCountBeforeBeginningAnySequence) {
  Fixture fixture;
  auto bad = plan();
  bad.sequence_count = 1;
  EXPECT_FALSE(fixture.driver.launch(bad, stage()).ok());
  EXPECT_EQ(fixture.first_ops.copies, 0U);
  EXPECT_EQ(fixture.second_ops.copies, 0U);
}

TEST(DeepSeekAttentionPlanComputeDriverTest,
     RuntimeBeginFailureStillDrainsAnEarlierSequence) {
  Fixture fixture;
  fixture.second_ops.fail_copy = true;
  ASSERT_TRUE(fixture.driver.launch(plan(), stage()).ok());
  EXPECT_EQ(fixture.first.state(),
            DeepSeekAttentionSequenceTransactionState::kAwaitingCompletion);
  EXPECT_EQ(fixture.second.state(),
            DeepSeekAttentionSequenceTransactionState::kPoisoned);
  EXPECT_EQ(fixture.driver.poll().value(), DeepSeekStageComputeStatus::kError);
  EXPECT_EQ(fixture.first.state(),
            DeepSeekAttentionSequenceTransactionState::kIdle);
  EXPECT_EQ(fixture.first_banks.generation(), 1U);
}

} }  // namespace pih
