#include "pih/model/deepseek_stage_compute_driver.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace pih {
namespace {

class RecordingStageBackend final : public DeepSeekStageOperatorBackend {
 public:
  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& plan) override {
    if (fail_launch_at == launches.size()) {
      return Status::Internal("injected operator launch failure");
    }
    EXPECT_EQ(plan.plan_sequence, 4U);
    launches.push_back(command);
    return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    return fail_poll ? DeepSeekStageComputeStatus::kError
                     : DeepSeekStageComputeStatus::kSuccess;
  }
  std::size_t fail_launch_at = static_cast<std::size_t>(-1);
  bool fail_poll = false;
  std::vector<DeepSeekStageOperatorCommand> launches;
};

DeepSeekPipelinePlanDescriptor decode_plan() {
  return {2, 4, DeepSeekPlanPhase::kDecode, 3, 3};
}

void run_to_completion(DeepSeekModelStageComputeDriver& driver) {
  for (int step = 0; step < 128; ++step) {
    auto status = driver.poll();
    ASSERT_TRUE(status.ok()) << status.status().message();
    if (*status == DeepSeekStageComputeStatus::kSuccess) return;
  }
  FAIL() << "stage compute did not complete";
}

TEST(DeepSeekStageComputeDriverTest, ExecutesOwnedLayersInReferenceOrder) {
  RecordingStageBackend backend;
  DeepSeekModelStageComputeDriver driver(backend);
  DeepSeekStagePlan stage{1, {14, 28}, false, false, false};
  ASSERT_TRUE(driver.launch(decode_plan(), stage).ok());
  run_to_completion(driver);
  ASSERT_EQ(backend.launches.size(), 30U);
  for (std::uint32_t index = 0; index < 15; ++index) {
    EXPECT_EQ(backend.launches[index * 2],
              (DeepSeekStageOperatorCommand{
                  DeepSeekStageOperatorKind::kAttention, 14 + index}));
    EXPECT_EQ(backend.launches[index * 2 + 1],
              (DeepSeekStageOperatorCommand{
                  DeepSeekStageOperatorKind::kMoe, 14 + index}));
  }
}

TEST(DeepSeekStageComputeDriverTest, OwnsEmbeddingHeadAndDsparkExactlyOnce) {
  auto plan = DeepSeekPipelinePlan::Create(1, true);
  ASSERT_TRUE(plan.ok());
  RecordingStageBackend backend;
  DeepSeekModelStageComputeDriver driver(backend);
  ASSERT_TRUE(driver.launch(decode_plan(), plan->rank(0)).ok());
  run_to_completion(driver);
  ASSERT_EQ(backend.launches.size(), 89U);
  EXPECT_EQ(backend.launches.front().kind,
            DeepSeekStageOperatorKind::kEmbedding);
  EXPECT_EQ(backend.launches[1],
            (DeepSeekStageOperatorCommand{
                DeepSeekStageOperatorKind::kAttention, 0}));
  EXPECT_EQ(backend.launches[87].kind, DeepSeekStageOperatorKind::kHead);
  EXPECT_EQ(backend.launches[88].kind, DeepSeekStageOperatorKind::kDspark);
}

TEST(DeepSeekStageComputeDriverTest, VerifyRunsMainModelWithoutDrafting) {
  auto plan = DeepSeekPipelinePlan::Create(2, true);
  ASSERT_TRUE(plan.ok());
  RecordingStageBackend backend;
  DeepSeekModelStageComputeDriver driver(backend);
  auto descriptor = decode_plan();
  descriptor.phase = DeepSeekPlanPhase::kVerify;
  ASSERT_TRUE(driver.launch(descriptor, plan->rank(1)).ok());
  run_to_completion(driver);
  EXPECT_EQ(std::count_if(backend.launches.begin(), backend.launches.end(),
                          [](const auto& command) {
                            return command.kind ==
                                   DeepSeekStageOperatorKind::kDspark;
                          }),
            0);
  EXPECT_EQ(backend.launches.back().kind, DeepSeekStageOperatorKind::kHead);
}

TEST(DeepSeekStageComputeDriverTest,
     DsparkEnabledPrefillRunsStateInitializationAfterMainModel) {
  auto pipeline = DeepSeekPipelinePlan::Create(2, true).value();
  RecordingStageBackend backend;
  DeepSeekModelStageComputeDriver driver(backend);
  auto descriptor = decode_plan();
  descriptor.phase = DeepSeekPlanPhase::kPrefill;
  ASSERT_TRUE(driver.launch(descriptor, pipeline.rank(1)).ok());
  run_to_completion(driver);
  ASSERT_FALSE(backend.launches.empty());
  EXPECT_EQ(backend.launches.back().kind,
            DeepSeekStageOperatorKind::kDspark);
  EXPECT_EQ(std::count_if(backend.launches.begin(), backend.launches.end(),
                          [](const auto& command) {
                            return command.kind ==
                                   DeepSeekStageOperatorKind::kDspark;
                          }),
            1);
}

TEST(DeepSeekStageComputeDriverTest, DrainDoesNotLaunchModelOperators) {
  RecordingStageBackend backend;
  DeepSeekModelStageComputeDriver driver(backend);
  auto descriptor = decode_plan();
  descriptor.phase = DeepSeekPlanPhase::kDrain;
  descriptor.token_count = 0;
  ASSERT_TRUE(driver.launch(descriptor, {0, {0, 42}, true, true, false}).ok());
  auto status = driver.poll();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(*status, DeepSeekStageComputeStatus::kSuccess);
  EXPECT_TRUE(backend.launches.empty());
}

TEST(DeepSeekStageComputeDriverTest, OperatorFailureIsFailStop) {
  RecordingStageBackend backend;
  backend.fail_poll = true;
  DeepSeekModelStageComputeDriver driver(backend);
  ASSERT_TRUE(driver.launch(decode_plan(), {0, {0, 0}, true, false, false}).ok());
  auto status = driver.poll();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(*status, DeepSeekStageComputeStatus::kError);
  EXPECT_FALSE(driver.poll().ok());
}

TEST(DeepSeekStageComputeDriverTest, SuccessfulDriverIsReusableForNextPlan) {
  RecordingStageBackend backend;
  DeepSeekModelStageComputeDriver driver(backend);
  const DeepSeekStagePlan stage{0, {0, 0}, true, false, false};
  ASSERT_TRUE(driver.launch(decode_plan(), stage).ok());
  run_to_completion(driver);
  ASSERT_TRUE(driver.launch(decode_plan(), stage).ok());
  run_to_completion(driver);
  EXPECT_EQ(backend.launches.size(), 6U);
}

}  // namespace
}  // namespace pih
