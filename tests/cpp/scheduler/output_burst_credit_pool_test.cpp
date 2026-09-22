#include "pih/scheduler/output_burst_credit_pool.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(OutputBurstCreditPoolTest, EnforcesPrepareCommitTransferReleaseLifecycle) {
  auto pool = OutputBurstCreditPool::Create({1, 1, 5, 8, 4096}).value();
  auto lease = pool.acquire(7, OutputPlanKind::kDeepSeek, 5, 8, 4096).value();
  EXPECT_EQ(pool.available(), 0U);
  EXPECT_TRUE(pool.validate_abort(lease).ok());
  EXPECT_FALSE(pool.validate_transfer(lease).ok());
  EXPECT_FALSE(pool.release(lease).ok());
  ASSERT_TRUE(pool.commit(lease).ok());
  EXPECT_FALSE(pool.validate_abort(lease).ok());
  EXPECT_TRUE(pool.validate_transfer(lease).ok());
  EXPECT_FALSE(pool.abort(lease).ok());
  ASSERT_TRUE(pool.transfer(lease).ok());
  ASSERT_TRUE(pool.release(lease).ok());
  EXPECT_EQ(pool.available(), 1U);
}

TEST(OutputBurstCreditPoolTest, OneShortAndForgedLeasesDoNotMutateCredit) {
  auto pool = OutputBurstCreditPool::Create({1, 1, 5, 8, 4096}).value();
  auto lease = pool.acquire(7, OutputPlanKind::kQwen, 1, 8, 4096).value();
  EXPECT_FALSE(pool.acquire(8, OutputPlanKind::kQwen, 1, 1, 1).ok());
  auto forged = lease;
  --forged.bytes;
  EXPECT_FALSE(pool.validate_commit(forged).ok());
  EXPECT_FALSE(pool.commit(forged).ok());
  EXPECT_TRUE(pool.validate_commit(lease).ok());
  EXPECT_TRUE(pool.abort(lease).ok());
  EXPECT_EQ(pool.available(), 1U);
}

TEST(OutputBurstCreditPoolTest, RejectsPlanEnvelopeOverflow) {
  auto pool = OutputBurstCreditPool::Create({1, 1, 5, 8, 4096}).value();
  EXPECT_FALSE(pool.acquire(1, OutputPlanKind::kQwen, 2, 1, 1).ok());
  EXPECT_FALSE(pool.acquire(1, OutputPlanKind::kDeepSeek, 6, 1, 1).ok());
  EXPECT_FALSE(OutputBurstCreditPool::Create({1, 2, 5, 8, 4096}).ok());
}

}  // namespace
}  // namespace pih
