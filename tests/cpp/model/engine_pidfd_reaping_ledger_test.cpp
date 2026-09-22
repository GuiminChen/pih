#include "pih/model/engine_pidfd_reaping_ledger.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

std::vector<EngineSupervisedProcessIdentity> processes(std::uint32_t ranks = 2) {
  std::vector<EngineSupervisedProcessIdentity> values{
      {EngineSupervisedProcessRole::kController, -1, 100, 200}};
  for (std::uint32_t rank = 0; rank < ranks; ++rank)
    values.push_back({EngineSupervisedProcessRole::kRank,
                      static_cast<std::int32_t>(rank), 101 + rank, 201 + rank});
  return values;
}

EnginePidfdReapedReceipt receipt(
    std::uint64_t event, EngineSupervisedProcessIdentity process) {
  return {7, event, process, true, true};
}

TEST(EnginePidfdReapingLedgerTest, CompletesControllerAndOneToFourRanks) {
  for (std::uint32_t ranks = 1; ranks <= 4; ++ranks) {
    const auto expected = processes(ranks);
    auto ledger = EnginePidfdReapingLedger::Create(7, expected).value();
    EXPECT_EQ(ledger.remaining(), ranks + 1);
    for (std::size_t index = 0; index < expected.size(); ++index)
      ASSERT_TRUE(ledger.accept(receipt(index + 1, expected[index])).ok());
    EXPECT_TRUE(ledger.complete());
    EXPECT_EQ(ledger.remaining(), 0U);
  }
}

TEST(EnginePidfdReapingLedgerTest, RejectsInvalidManifest) {
  EXPECT_FALSE(EnginePidfdReapingLedger::Create(0, processes()).ok());
  EXPECT_FALSE(EnginePidfdReapingLedger::Create(7, {}).ok());
  auto value = processes(); value.erase(value.begin());
  EXPECT_FALSE(EnginePidfdReapingLedger::Create(7, value).ok());
  value = processes(); value[2].rank = 0;
  EXPECT_FALSE(EnginePidfdReapingLedger::Create(7, value).ok());
  value = processes(); value[2].pidfd_identity = value[1].pidfd_identity;
  EXPECT_FALSE(EnginePidfdReapingLedger::Create(7, value).ok());
}

TEST(EnginePidfdReapingLedgerTest, IdentityDriftReplayAndGapPoison) {
  for (int mutation = 0; mutation < 6; ++mutation) {
    const auto expected = processes();
    auto ledger = EnginePidfdReapingLedger::Create(7, expected).value();
    auto value = receipt(1, expected[0]);
    if (mutation == 0) value.engine_generation = 8;
    if (mutation == 1) value.event_identity = 2;
    if (mutation == 2) ++value.process.process_identity;
    if (mutation == 3) ++value.process.pidfd_identity;
    if (mutation == 4) value.exited = false;
    if (mutation == 5) value.reaped = false;
    EXPECT_FALSE(ledger.accept(value).ok()) << mutation;
    EXPECT_TRUE(ledger.poisoned()) << mutation;
  }
  const auto expected = processes();
  auto ledger = EnginePidfdReapingLedger::Create(7, expected).value();
  ASSERT_TRUE(ledger.accept(receipt(1, expected[0])).ok());
  auto replay = receipt(2, expected[0]);
  EXPECT_FALSE(ledger.accept(replay).ok());
  EXPECT_TRUE(ledger.poisoned());
}

}  // namespace
}  // namespace pih
