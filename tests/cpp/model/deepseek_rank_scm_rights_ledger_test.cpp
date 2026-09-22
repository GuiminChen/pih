#include "pih/model/deepseek_rank_scm_rights_ledger.h"

#include <gtest/gtest.h>

#include <limits>
#include <optional>

namespace pih {
namespace {

TEST(DeepSeekRankScmRightsLedgerTest,
     TracksNestedMoveOnlyLeasesAndReleasesExactlyOnce) {
  EXPECT_EQ(kDeepSeekRankScmRightsLedgerAbi,
            "pih_deepseek_rank_scm_rights_ledger_v1");
  DeepSeekRankScmRightsInFlightLedger ledger;
  EXPECT_EQ(ledger.sample_inflight_fd_count().value(), 0U);
  auto first = ledger.acquire(2).value();
  EXPECT_EQ(ledger.sample_inflight_fd_count().value(), 2U);
  {
    auto second = ledger.acquire(3).value();
    EXPECT_EQ(ledger.sample_inflight_fd_count().value(), 5U);
    auto moved = std::move(second);
    EXPECT_FALSE(second.active());
    EXPECT_TRUE(moved.active());
    EXPECT_EQ(moved.descriptor_count(), 3U);
  }
  EXPECT_EQ(ledger.sample_inflight_fd_count().value(), 2U);
  first = ledger.acquire(1).value();
  EXPECT_EQ(ledger.sample_inflight_fd_count().value(), 1U);
}

TEST(DeepSeekRankScmRightsLedgerTest, RejectsEmptyAndOverflowingLease) {
  DeepSeekRankScmRightsInFlightLedger ledger;
  EXPECT_EQ(ledger.acquire(0).status().code(), StatusCode::kInvalidArgument);
  auto maximum = ledger.acquire(
      std::numeric_limits<std::uint64_t>::max()).value();
  auto overflow = ledger.acquire(1);
  EXPECT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.status().code(), StatusCode::kResourceExhausted);
  EXPECT_EQ(ledger.sample_inflight_fd_count().value(),
            std::numeric_limits<std::uint64_t>::max());
}

}  // namespace
}  // namespace pih
