#include "pih/model/deepseek_expert_compute_driver.h"

#include <gtest/gtest.h>

#include <deque>
#include <vector>

namespace pih {
namespace {

class FakeComputeBackend final : public DeepSeekExpertComputeBackend {
 public:
  Status submit(const DeepSeekExpertComputeSubmission& submission) override {
    if (!submit_status.ok()) return submit_status;
    last = submission;
    copied_routes.assign(submission.routes,
                         submission.routes + submission.route_count);
    ++submissions;
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    if (polls.empty()) return DeepSeekExpertAsyncStatus::kInProgress;
    auto value = polls.front();
    polls.pop_front();
    return value;
  }

  Status submit_status = Status::Ok();
  DeepSeekExpertComputeSubmission last{};
  std::vector<DeepSeekExpertRoute> copied_routes;
  std::deque<DeepSeekExpertAsyncStatus> polls;
  std::uint32_t submissions = 0;
};

Result<DeepSeekExpertComputeDriver> make_driver(FakeComputeBackend& backend) {
  auto table = DeepSeekExpertSlotTable::Create(
      {0x10000000U, 0x11000000U}, 77, 0);
  if (!table.ok()) return table.status();
  auto layout = DeepSeekExpertComputeArenaLayout::Create(4);
  if (!layout.ok()) return layout.status();
  auto arena = layout->bind(0x20000000U, layout->required_bytes());
  if (!arena.ok()) return arena.status();
  return DeepSeekExpertComputeDriver::Create(
      std::move(*table), *arena, 4, 0x30000000U, 0x40000000U,
      0x50000000U, backend);
}

TEST(DeepSeekExpertComputeDriverTest, SubmitsLeaseAndCanonicalRouteSlice) {
  FakeComputeBackend backend;
  auto driver = make_driver(backend);
  ASSERT_TRUE(driver.ok());
  const std::vector<DeepSeekExpertRoute> routes = {
      {3, 0, 1, 0.25F}, {3, 2, 0, 0.75F}};
  EXPECT_TRUE(driver->launch({{2, 3}, 1, 9}, routes.data(), 2).ok());
  EXPECT_EQ(backend.submissions, 1U);
  EXPECT_EQ(backend.last.expert.identity,
            (DeepSeekExpertIdentity{2, 3}));
  EXPECT_EQ(backend.last.expert.generation, 9U);
  EXPECT_EQ(backend.last.expert.bundle.w1.packed.address, 0x11000000U);
  EXPECT_EQ(backend.last.route_count, 2U);
  EXPECT_EQ(backend.copied_routes, routes);
  EXPECT_EQ(backend.last.packed_token_count, 4U);
  EXPECT_EQ(backend.last.source_hidden_bf16, 0x30000000U);
  EXPECT_EQ(backend.last.accumulator_f32, 0x40000000U);
}

TEST(DeepSeekExpertComputeDriverTest, EnforcesSingleInflightLane) {
  FakeComputeBackend backend;
  auto driver = make_driver(backend);
  ASSERT_TRUE(driver.ok());
  const std::vector<DeepSeekExpertRoute> routes = {{3, 0, 0, 1.0F}};
  ASSERT_TRUE(driver->launch({{2, 3}, 0, 1}, routes.data(), 1).ok());
  EXPECT_FALSE(driver->launch({{2, 3}, 0, 1}, routes.data(), 1).ok());
  backend.polls = {DeepSeekExpertAsyncStatus::kInProgress,
                   DeepSeekExpertAsyncStatus::kSuccess};
  EXPECT_EQ(driver->poll().value(), DeepSeekExpertAsyncStatus::kInProgress);
  EXPECT_EQ(driver->poll().value(), DeepSeekExpertAsyncStatus::kSuccess);
  EXPECT_TRUE(driver->launch({{2, 3}, 0, 1}, routes.data(), 1).ok());
}

TEST(DeepSeekExpertComputeDriverTest,
     SubmitsFixedResidentMainBundleWithoutPagerLease) {
  FakeComputeBackend backend;
  auto driver = make_driver(backend);
  ASSERT_TRUE(driver.ok());
  const DeepSeekExpertBundleDeviceView bundle{
      {{0x61000000U, 4194304}, {0x62000000U, 262144}},
      {{0x63000000U, 4194304}, {0x64000000U, 262144}},
      {{0x65000000U, 4194304}, {0x66000000U, 262144}}};
  const std::vector<DeepSeekExpertRoute> routes = {{7, 0, 0, 1.0F}};
  ASSERT_TRUE(driver->launch_resident(
      {42, 7}, 17, bundle, routes.data(), 1).ok());
  EXPECT_EQ(backend.last.expert.identity,
            (DeepSeekExpertIdentity{42, 7}));
  EXPECT_EQ(backend.last.expert.slot, UINT32_MAX);
  EXPECT_EQ(backend.last.expert.bundle.w2.packed.address, 0x63000000U);
}

TEST(DeepSeekExpertComputeDriverTest, RejectsNoncanonicalRoutesBeforeSubmit) {
  FakeComputeBackend backend;
  auto driver = make_driver(backend);
  ASSERT_TRUE(driver.ok());
  const std::vector<DeepSeekExpertRoute> wrong_expert = {{4, 0, 0, 1.0F}};
  EXPECT_FALSE(driver->launch({{2, 3}, 0, 1}, wrong_expert.data(), 1).ok());
  const std::vector<DeepSeekExpertRoute> reverse = {
      {3, 2, 0, 0.5F}, {3, 1, 1, 0.5F}};
  EXPECT_FALSE(driver->launch({{2, 3}, 0, 1}, reverse.data(), 2).ok());
  EXPECT_EQ(backend.submissions, 0U);
}

TEST(DeepSeekExpertComputeDriverTest, BackendFailurePoisonsDriver) {
  FakeComputeBackend backend;
  auto driver = make_driver(backend);
  ASSERT_TRUE(driver.ok());
  const std::vector<DeepSeekExpertRoute> routes = {{3, 0, 0, 1.0F}};
  ASSERT_TRUE(driver->launch({{2, 3}, 0, 1}, routes.data(), 1).ok());
  backend.polls = {DeepSeekExpertAsyncStatus::kError};
  EXPECT_EQ(driver->poll().value(), DeepSeekExpertAsyncStatus::kError);
  EXPECT_FALSE(driver->launch({{2, 3}, 0, 2}, routes.data(), 1).ok());
}

}  // namespace
}  // namespace pih
