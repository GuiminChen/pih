#include "pih/model/deepseek_expert_subwave_executor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

namespace pih {
namespace {

std::vector<DeepSeekExpertRoute> six_routes() {
  std::vector<DeepSeekExpertRoute> routes;
  for (std::uint16_t expert : {11, 3, 9, 1, 7, 5}) {
    routes.push_back({expert, 0, static_cast<std::uint8_t>(routes.size()),
                      1.0F / 6.0F});
  }
  return routes;
}

class ImmediateTransfer final : public DeepSeekExpertTransferDriver {
 public:
  Status start(DeepSeekExpertIdentity identity, std::uint32_t,
               std::uint64_t generation, std::uint64_t payload_bytes) override {
    if (fail_expert.has_value() && identity.expert == *fail_expert) {
      return Status::Internal("injected transfer failure");
    }
    EXPECT_EQ(payload_bytes, DeepSeekExpertPager::kBundleBytes);
    starts.emplace_back(identity.expert, generation);
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> poll(
      DeepSeekExpertIdentity, std::uint64_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  std::optional<std::uint16_t> fail_expert;
  std::vector<std::pair<std::uint16_t, std::uint64_t>> starts;
};

class ImmediateKernel final : public DeepSeekExpertKernelDriver {
 public:
  Status launch(const DeepSeekExpertLease& lease,
                const DeepSeekExpertRoute* routes,
                std::uint32_t route_count) override {
    EXPECT_EQ(route_count, 1U);
    EXPECT_EQ(routes[0].expert_id, lease.identity.expert);
    launches.push_back(lease.identity.expert);
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return fail_poll ? DeepSeekExpertAsyncStatus::kError
                     : DeepSeekExpertAsyncStatus::kSuccess;
  }
  bool fail_poll = false;
  std::vector<std::uint16_t> launches;
};

TEST(DeepSeekExpertSubwaveExecutorTest, ExecutesSixMissesWithOnlyTwoSlots) {
  auto plan = DeepSeekExpertSubwavePlan::Create(1, six_routes());
  auto pager = DeepSeekExpertPager::Create({4, 4}, 2, 2);
  ASSERT_TRUE(plan.ok()); ASSERT_TRUE(pager.ok());
  auto executor = DeepSeekExpertSubwaveExecutor::Create(4, *plan, *pager);
  ASSERT_TRUE(executor.ok());
  ImmediateTransfer transfer;
  ImmediateKernel kernel;
  for (int step = 0; step < 16 &&
       executor->state() != DeepSeekExpertSubwaveExecutorState::kComplete;
       ++step) {
    const auto status = executor->advance(transfer, kernel);
    ASSERT_TRUE(status.ok() || status.code() == StatusCode::kUnavailable)
        << status.message();
  }
  EXPECT_EQ(executor->state(), DeepSeekExpertSubwaveExecutorState::kComplete);
  EXPECT_EQ(executor->completed_experts(), 6U);
  EXPECT_EQ(kernel.launches,
            (std::vector<std::uint16_t>{1, 3, 5, 7, 9, 11}));
  EXPECT_EQ(transfer.starts.size(), 6U);
  EXPECT_FALSE(pager->poisoned());
}

TEST(DeepSeekExpertSubwaveExecutorTest, ReusesResidentExpertWithoutTransfer) {
  auto plan = DeepSeekExpertSubwavePlan::Create(1, six_routes());
  auto pager = DeepSeekExpertPager::Create({4, 4}, 6, 2);
  ASSERT_TRUE(plan.ok()); ASSERT_TRUE(pager.ok());
  auto resident = pager->demand({4, 1});
  ASSERT_TRUE(resident.ok());
  ASSERT_TRUE(pager->begin_h2d({4, 1}, resident->generation,
                               DeepSeekExpertPager::kBundleBytes).ok());
  ASSERT_TRUE(pager->complete_h2d({4, 1}, resident->generation).ok());
  auto executor = DeepSeekExpertSubwaveExecutor::Create(4, *plan, *pager);
  ASSERT_TRUE(executor.ok());
  ImmediateTransfer transfer;
  ImmediateKernel kernel;
  for (int step = 0; step < 16 &&
       executor->state() != DeepSeekExpertSubwaveExecutorState::kComplete;
       ++step) {
    const auto status = executor->advance(transfer, kernel);
    ASSERT_TRUE(status.ok() || status.code() == StatusCode::kUnavailable);
  }
  EXPECT_EQ(transfer.starts.size(), 5U);
  EXPECT_EQ(kernel.launches.front(), 1U);
}

TEST(DeepSeekExpertSubwaveExecutorTest, TransferFailurePoisonsPagerAndExecutor) {
  auto plan = DeepSeekExpertSubwavePlan::Create(1, six_routes());
  auto pager = DeepSeekExpertPager::Create({4, 4}, 2, 2);
  ASSERT_TRUE(plan.ok()); ASSERT_TRUE(pager.ok());
  auto executor = DeepSeekExpertSubwaveExecutor::Create(4, *plan, *pager);
  ASSERT_TRUE(executor.ok());
  ImmediateTransfer transfer;
  transfer.fail_expert = 1;
  ImmediateKernel kernel;
  EXPECT_FALSE(executor->advance(transfer, kernel).ok());
  EXPECT_EQ(executor->state(), DeepSeekExpertSubwaveExecutorState::kPoisoned);
  EXPECT_TRUE(pager->poisoned());
}

TEST(DeepSeekExpertSubwaveExecutorTest, RetriesSameExpertAfterSlotBackpressure) {
  auto plan = DeepSeekExpertSubwavePlan::Create(1, six_routes());
  auto pager = DeepSeekExpertPager::Create({4, 4}, 2, 2);
  ASSERT_TRUE(plan.ok()); ASSERT_TRUE(pager.ok());
  auto first = pager->demand({4, 200});
  auto second = pager->demand({4, 201});
  ASSERT_TRUE(first.ok()); ASSERT_TRUE(second.ok());
  ASSERT_TRUE(pager->begin_h2d({4, 200}, first->generation,
                               DeepSeekExpertPager::kBundleBytes).ok());
  ASSERT_TRUE(pager->complete_h2d({4, 200}, first->generation).ok());
  ASSERT_TRUE(pager->begin_h2d({4, 201}, second->generation,
                               DeepSeekExpertPager::kBundleBytes).ok());
  ASSERT_TRUE(pager->complete_h2d({4, 201}, second->generation).ok());
  auto executor = DeepSeekExpertSubwaveExecutor::Create(4, *plan, *pager);
  ASSERT_TRUE(executor.ok());
  ImmediateTransfer transfer;
  ImmediateKernel kernel;
  EXPECT_EQ(executor->advance(transfer, kernel).code(),
            StatusCode::kUnavailable);
  EXPECT_TRUE(pager->request_eviction({4, 200}, first->generation).ok());
  ASSERT_TRUE(executor->advance(transfer, kernel).ok());
  EXPECT_EQ(kernel.launches.front(), 1U);
}

TEST(DeepSeekExpertSubwaveExecutorTest, KernelFailurePoisonsPagerAndExecutor) {
  auto plan = DeepSeekExpertSubwavePlan::Create(1, six_routes());
  auto pager = DeepSeekExpertPager::Create({4, 4}, 2, 2);
  ASSERT_TRUE(plan.ok()); ASSERT_TRUE(pager.ok());
  auto executor = DeepSeekExpertSubwaveExecutor::Create(4, *plan, *pager);
  ASSERT_TRUE(executor.ok());
  ImmediateTransfer transfer;
  ImmediateKernel kernel;
  kernel.fail_poll = true;
  ASSERT_TRUE(executor->advance(transfer, kernel).ok());
  EXPECT_FALSE(executor->advance(transfer, kernel).ok());
  EXPECT_EQ(executor->state(), DeepSeekExpertSubwaveExecutorState::kPoisoned);
  EXPECT_TRUE(pager->poisoned());
}

}  // namespace
}  // namespace pih
