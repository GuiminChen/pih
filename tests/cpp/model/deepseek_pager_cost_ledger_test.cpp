#include "pih/model/deepseek_pager_cost_ledger.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekPagerCostKey key(std::uint16_t expert = 7) {
  return {11, 19, 23, 3, expert};
}

TEST(DeepSeekPagerCostLedgerTest,
     DebitsOnlyFirstObservedMissAndNeverDebitsHits) {
  auto ledger = DeepSeekPagerCostLedger::Create(8);
  ASSERT_TRUE(ledger.ok());
  EXPECT_EQ(ledger->debit(key(), DeepSeekExpertDemandDisposition::kResidentHit,
                          0).value(),
            0U);
  EXPECT_EQ(ledger->debit(key(), DeepSeekExpertDemandDisposition::kNewMiss,
                          4000).value(),
            4000U);
  EXPECT_EQ(ledger->debit(key(), DeepSeekExpertDemandDisposition::kNewMiss,
                          4000).value(),
            0U);
  EXPECT_EQ(ledger->entry_count(), 1U);
  EXPECT_EQ(ledger->total_debit_ns(), 4000U);
  EXPECT_EQ(ledger->identity().hex().size(), 64U);
}

TEST(DeepSeekPagerCostLedgerTest,
     RejectsReplayDriftOverflowAndCapacityGrowth) {
  auto ledger = DeepSeekPagerCostLedger::Create(1).value();
  ASSERT_TRUE(ledger.debit(key(), DeepSeekExpertDemandDisposition::kNewMiss,
                           10).ok());
  EXPECT_FALSE(ledger.debit(key(), DeepSeekExpertDemandDisposition::kNewMiss,
                            11).ok());
  EXPECT_EQ(ledger.debit(key(8), DeepSeekExpertDemandDisposition::kNewMiss,
                         10).status().code(),
            StatusCode::kResourceExhausted);
  EXPECT_FALSE(DeepSeekPagerCostLedger::Create(0).ok());
  EXPECT_FALSE(DeepSeekPagerCostLedger::Create(
      DeepSeekPagerCostLedger::kMaximumEntries + 1).ok());
}

TEST(DeepSeekPagerCostLedgerTest,
     PagerCostObservationPreservesOriginalExpertDemandOrder) {
  auto pager = DeepSeekExpertPager::Create({0, 10}, 2, 2).value();
  auto ledger = DeepSeekPagerCostLedger::Create(2).value();
  const std::array<DeepSeekExpertIdentity, 2> route{{{3, 9}, {3, 2}}};
  auto first = pager.demand(route[0]).value();
  auto second = pager.demand(route[1]).value();
  EXPECT_EQ(first.slot, 0U);
  EXPECT_EQ(second.slot, 1U);
  EXPECT_EQ(pager.debit_observed_cost(
                ledger, {1, 1, 1, 3, 9}, first, 100).value(),
            100U);
  EXPECT_EQ(pager.debit_observed_cost(
                ledger, {1, 1, 1, 3, 2}, second, 200).value(),
            200U);
  EXPECT_EQ(first.slot, 0U);
  EXPECT_EQ(second.slot, 1U);
}

}  // namespace
}  // namespace pih
