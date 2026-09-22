#include "pih/model/deepseek_dspark_expert_compute_driver.h"

#include <gtest/gtest.h>

#include <deque>
#include <vector>

namespace pih {
namespace {

class Backend final : public DeepSeekExpertComputeBackend {
 public:
  Status submit(const DeepSeekExpertComputeSubmission& value) override {
    last = value;
    routes.assign(value.routes, value.routes + value.route_count);
    ++submissions;
    return submit_status;
  }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    if (polls.empty()) return DeepSeekExpertAsyncStatus::kInProgress;
    const auto value = polls.front();
    polls.pop_front();
    return value;
  }
  DeepSeekExpertComputeSubmission last;
  std::vector<DeepSeekExpertRoute> routes;
  std::deque<DeepSeekExpertAsyncStatus> polls;
  Status submit_status = Status::Ok();
  std::uint32_t submissions = 0;
};

DeepSeekExpertBundleDeviceView bundle() {
  return {
      {{0x61000000U, DeepSeekExpertBundleLayout::kPackedBytesPerMatrix},
       {0x62000000U, DeepSeekExpertBundleLayout::kScaleBytesPerMatrix}},
      {{0x63000000U, DeepSeekExpertBundleLayout::kPackedBytesPerMatrix},
       {0x64000000U, DeepSeekExpertBundleLayout::kScaleBytesPerMatrix}},
      {{0x65000000U, DeepSeekExpertBundleLayout::kPackedBytesPerMatrix},
       {0x66000000U, DeepSeekExpertBundleLayout::kScaleBytesPerMatrix}}};
}

Result<DeepSeekDsparkExpertComputeDriver> driver(Backend& backend) {
  auto layout = DeepSeekExpertComputeArenaLayout::Create(5);
  if (!layout.ok()) return layout.status();
  auto arena = layout->bind(0x20000000U, layout->required_bytes());
  if (!arena.ok()) return arena.status();
  return DeepSeekDsparkExpertComputeDriver::Create(
      *arena, 0x30000000U, 0x40000000U, 0x50000000U, 77, backend);
}

TEST(DeepSeekDsparkExpertComputeDriverTest,
     SubmitsStrongStageWithoutSyntheticMainLayer) {
  Backend backend;
  auto value = driver(backend).value();
  const std::vector<DeepSeekExpertRoute> routes{
      {7, 0, 0, 0.25F}, {7, 3, 1, 0.75F}};
  ASSERT_TRUE(value.launch_resident(
      DeepSeekDsparkStageId::kMtp2, 7, 29, bundle(),
      routes.data(), static_cast<std::uint32_t>(routes.size())).ok());
  EXPECT_EQ(backend.submissions, 1U);
  EXPECT_EQ(backend.last.context_identity, 77U);
  EXPECT_EQ(backend.last.expert.identity,
            (DeepSeekExpertIdentity{}));
  EXPECT_EQ(backend.last.bundle.w2.packed.address, 0x63000000U);
  EXPECT_EQ(backend.last.packed_token_count, 5U);
  EXPECT_EQ(backend.routes, routes);
  backend.polls = {DeepSeekExpertAsyncStatus::kSuccess};
  EXPECT_EQ(value.poll().value(), DeepSeekExpertAsyncStatus::kSuccess);
}

TEST(DeepSeekDsparkExpertComputeDriverTest,
     RejectsCrossExpertAndNoncanonicalRouteSlices) {
  Backend backend;
  auto value = driver(backend).value();
  std::vector<DeepSeekExpertRoute> routes{{8, 0, 0, 1.0F}};
  EXPECT_FALSE(value.launch_resident(
      DeepSeekDsparkStageId::kMtp0, 7, 29, bundle(),
      routes.data(), 1).ok());
  routes = {{7, 4, 0, 0.5F}, {7, 1, 1, 0.5F}};
  EXPECT_FALSE(value.launch_resident(
      DeepSeekDsparkStageId::kMtp0, 7, 29, bundle(),
      routes.data(), 2).ok());
  EXPECT_EQ(backend.submissions, 0U);
}

TEST(DeepSeekDsparkExpertComputeDriverTest,
     BackendFailurePoisonsOnlyThisStageLane) {
  Backend backend;
  auto value = driver(backend).value();
  const DeepSeekExpertRoute route{7, 0, 0, 1.0F};
  ASSERT_TRUE(value.launch_resident(
      DeepSeekDsparkStageId::kMtp1, 7, 29, bundle(), &route, 1).ok());
  backend.polls = {DeepSeekExpertAsyncStatus::kError};
  EXPECT_EQ(value.poll().value(), DeepSeekExpertAsyncStatus::kError);
  EXPECT_FALSE(value.launch_resident(
      DeepSeekDsparkStageId::kMtp1, 7, 30, bundle(), &route, 1).ok());
}

}  // namespace
}  // namespace pih
