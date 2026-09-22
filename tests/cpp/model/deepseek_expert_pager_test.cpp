#include "pih/model/deepseek_expert_pager.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(DeepSeekExpertPagerTest, EnforcesCanonicalBundleAndStartupWindow) {
  EXPECT_EQ(DeepSeekExpertPager::kBundleBytes, 13369344ULL);
  EXPECT_FALSE(DeepSeekExpertPager::Create({0, 10}, 1, 2).ok());
  EXPECT_FALSE(DeepSeekExpertPager::Create({0, 10}, 2, 1).ok());
  EXPECT_TRUE(DeepSeekExpertPager::Create({0, 10}, 2, 2).ok());
  EXPECT_FALSE(DeepSeekExpertPager::Create({42, 43}, 2, 2).ok());
  EXPECT_FALSE(DeepSeekExpertPager::Create({42, 44}, 2, 2).ok());
}

TEST(DeepSeekExpertPagerTest, MergesInflightDemandAndPublishesLease) {
  auto pager = DeepSeekExpertPager::Create({0, 10}, 2, 2);
  ASSERT_TRUE(pager.ok());
  DeepSeekExpertIdentity expert{3, 17};
  auto miss = pager->demand(expert);
  ASSERT_TRUE(miss.ok());
  EXPECT_EQ(miss->disposition, DeepSeekExpertDemandDisposition::kNewMiss);
  EXPECT_EQ(pager->transfer_reservations(), 1U);
  auto joined = pager->demand(expert);
  ASSERT_TRUE(joined.ok());
  EXPECT_EQ(joined->disposition, DeepSeekExpertDemandDisposition::kInflightJoin);
  EXPECT_EQ(joined->slot, miss->slot);
  EXPECT_TRUE(pager->begin_h2d(expert, miss->generation,
                               DeepSeekExpertPager::kBundleBytes).ok());
  EXPECT_TRUE(pager->complete_h2d(expert, miss->generation).ok());
  auto lease = pager->acquire(expert, miss->generation);
  ASSERT_TRUE(lease.ok());
  EXPECT_EQ(lease->identity, expert);
  EXPECT_EQ(pager->transfer_reservations(), 0U);
  EXPECT_TRUE(pager->release(*lease).ok());
}

TEST(DeepSeekExpertPagerTest, PreventsStaleGenerationAndEarlyRebind) {
  auto pager = DeepSeekExpertPager::Create({0, 0}, 2, 2);
  ASSERT_TRUE(pager.ok());
  auto a = pager->demand({0, 1});
  auto b = pager->demand({0, 2});
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(pager->demand({0, 3}).status().code(), StatusCode::kResourceExhausted);
  ASSERT_TRUE(pager->begin_h2d({0, 1}, a->generation,
                               DeepSeekExpertPager::kBundleBytes).ok());
  ASSERT_TRUE(pager->complete_h2d({0, 1}, a->generation).ok());
  auto lease = pager->acquire({0, 1}, a->generation);
  ASSERT_TRUE(lease.ok());
  EXPECT_TRUE(pager->request_eviction({0, 1}, a->generation).ok());
  EXPECT_EQ(pager->demand({0, 3}).status().code(), StatusCode::kResourceExhausted);
  EXPECT_FALSE(pager->release({lease->identity, lease->slot,
                               lease->generation + 1}).ok());
  ASSERT_TRUE(pager->release(*lease).ok());
  auto c = pager->demand({0, 3});
  ASSERT_TRUE(c.ok());
  EXPECT_EQ(c->slot, a->slot);
  EXPECT_GT(c->generation, a->generation);
}

TEST(DeepSeekExpertPagerTest, RejectsForeignLayerAndPayloadMismatch) {
  auto pager = DeepSeekExpertPager::Create({11, 21}, 2, 2);
  ASSERT_TRUE(pager.ok());
  EXPECT_EQ(pager->demand({10, 1}).status().code(), StatusCode::kInvalidArgument);
  auto miss = pager->demand({11, 1});
  ASSERT_TRUE(miss.ok());
  EXPECT_EQ(pager->begin_h2d({11, 1}, miss->generation, 1).code(),
            StatusCode::kInvalidArgument);
}

TEST(DeepSeekExpertPagerTest, TransferFailurePoisonsWholeEpoch) {
  auto pager = DeepSeekExpertPager::Create({0, 42}, 2, 2);
  ASSERT_TRUE(pager.ok());
  auto miss = pager->demand({4, 9});
  ASSERT_TRUE(miss.ok());
  ASSERT_TRUE(pager->begin_h2d({4, 9}, miss->generation,
                               DeepSeekExpertPager::kBundleBytes).ok());
  ASSERT_TRUE(pager->fail_transfer({4, 9}, miss->generation).ok());
  EXPECT_TRUE(pager->poisoned());
  EXPECT_EQ(pager->transfer_reservations(), 0U);
  EXPECT_EQ(pager->demand({4, 10}).status().code(), StatusCode::kUnavailable);
  EXPECT_EQ(pager->complete_h2d({4, 9}, miss->generation).code(),
            StatusCode::kUnavailable);
  EXPECT_TRUE(pager->fail_transfer({4, 9}, miss->generation).ok());
}

}  // namespace
}  // namespace pih
