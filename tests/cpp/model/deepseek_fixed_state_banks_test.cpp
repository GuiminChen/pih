#include "pih/model/deepseek_fixed_state_banks.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class Operations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t destination, std::uintptr_t source,
                        std::uint64_t bytes, std::uintptr_t) override {
    copied_destination = destination; copied_source = source;
    copied_bytes = bytes; return copy_status;
  }
  Status record_event(std::uintptr_t event, std::uintptr_t) override {
    recorded_event = event; return record_status;
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return query_status;
  }
  std::uintptr_t copied_destination = 0;
  std::uintptr_t copied_source = 0;
  std::uint64_t copied_bytes = 0;
  std::uintptr_t recorded_event = 0;
  Status copy_status = Status::Ok();
  Status record_status = Status::Ok();
  DeepSeekExpertAsyncStatus query_status =
      DeepSeekExpertAsyncStatus::kSuccess;
};

TEST(DeepSeekFixedStateBanksTest, CommitsOnlyAfterCompletion) {
  Operations operations;
  auto banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 1024}, {0x2000, 1024}, 77, operations);
  ASSERT_TRUE(banks.ok());
  EXPECT_TRUE(banks->prepare(9).ok());
  EXPECT_EQ(operations.copied_source, 0x1000U);
  EXPECT_EQ(operations.copied_destination, 0x2000U);
  EXPECT_FALSE(banks->commit().ok());
  EXPECT_TRUE(banks->seal(9).ok());
  EXPECT_EQ(operations.recorded_event, 77U);
  EXPECT_EQ(banks->poll().value(), DeepSeekExpertAsyncStatus::kSuccess);
  EXPECT_TRUE(banks->commit().ok());
  EXPECT_EQ(banks->committed_address(), 0x2000U);
  EXPECT_EQ(banks->generation(), 2U);
}

TEST(DeepSeekFixedStateBanksTest, AbortKeepsCommittedGeneration) {
  Operations operations;
  auto banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 1024}, {0x2000, 1024}, 77, operations);
  ASSERT_TRUE(banks.ok());
  ASSERT_TRUE(banks->prepare(9).ok());
  ASSERT_TRUE(banks->seal(9).ok());
  ASSERT_TRUE(banks->poll().ok());
  EXPECT_TRUE(banks->abort().ok());
  EXPECT_EQ(banks->committed_address(), 0x1000U);
  EXPECT_EQ(banks->generation(), 1U);
  EXPECT_TRUE(banks->prepare(9).ok());
}

TEST(DeepSeekFixedStateBanksTest, DoesNotReuseBeforeEventCompletes) {
  Operations operations;
  operations.query_status = DeepSeekExpertAsyncStatus::kInProgress;
  auto banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 1024}, {0x2000, 1024}, 77, operations);
  ASSERT_TRUE(banks.ok());
  ASSERT_TRUE(banks->prepare(9).ok());
  ASSERT_TRUE(banks->seal(9).ok());
  EXPECT_EQ(banks->poll().value(), DeepSeekExpertAsyncStatus::kInProgress);
  EXPECT_FALSE(banks->abort().ok());
  EXPECT_FALSE(banks->prepare(9).ok());
}

TEST(DeepSeekFixedStateBanksTest, RejectsAliasedOrUnequalBanks) {
  Operations operations;
  EXPECT_FALSE(DeepSeekFixedStateBanks::Create(
                   {0x1000, 1024}, {0x1200, 1024}, 77, operations)
                   .ok());
  EXPECT_FALSE(DeepSeekFixedStateBanks::Create(
                   {0x1000, 1024}, {0x2000, 512}, 77, operations)
                   .ok());
}

} }  // namespace pih
