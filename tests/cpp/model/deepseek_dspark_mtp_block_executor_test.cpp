#include "pih/model/deepseek_dspark_mtp_block_executor.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih { namespace {
class Backend final : public DeepSeekDsparkMtpOperatorBackend {
 public:
  Status launch(const DeepSeekDsparkMtpOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor&) override {
    commands.push_back(command); return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    ++polls; return DeepSeekStageComputeStatus::kSuccess;
  }
  Status cancel() override { ++cancels; return Status::Ok(); }
  std::vector<DeepSeekDsparkMtpOperatorCommand> commands;
  std::uint32_t polls = 0;
  std::uint32_t cancels = 0;
};
DeepSeekPipelinePlanDescriptor plan(DeepSeekPlanPhase phase =
                                    DeepSeekPlanPhase::kDecode) {
  return {3, 7, phase, 2, 2};
}
TEST(DeepSeekDsparkMtpBlockExecutorTest,
     RunsAttentionThenMoeForThreeExplicitStageIdentities) {
  Backend backend;
  auto executor = DeepSeekDsparkMtpBlockExecutor::Create(backend).value();
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    const auto stage = deepseek_dspark_stage_id(index).value();
    ASSERT_TRUE(executor.launch(stage, plan()).ok());
    EXPECT_EQ(executor.poll().value(),
              DeepSeekStageComputeStatus::kInProgress);
    EXPECT_EQ(executor.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  }
  ASSERT_EQ(backend.commands.size(), 6U);
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    const auto stage = deepseek_dspark_stage_id(index).value();
    EXPECT_EQ(backend.commands[index * 2],
              (DeepSeekDsparkMtpOperatorCommand{
                  stage, DeepSeekDsparkMtpOperatorKind::kAttention}));
    EXPECT_EQ(backend.commands[index * 2 + 1],
              (DeepSeekDsparkMtpOperatorCommand{
                  stage, DeepSeekDsparkMtpOperatorKind::kMoe}));
  }
}
TEST(DeepSeekDsparkMtpBlockExecutorTest,
     PrefillRunsAttentionOnlyForAllThreeStages) {
  Backend backend;
  auto executor = DeepSeekDsparkMtpBlockExecutor::Create(backend).value();
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    const auto stage = deepseek_dspark_stage_id(index).value();
    ASSERT_TRUE(executor.launch(
        stage, plan(DeepSeekPlanPhase::kPrefill)).ok());
    EXPECT_EQ(executor.poll().value(),
              DeepSeekStageComputeStatus::kSuccess);
  }
  ASSERT_EQ(backend.commands.size(), kDeepSeekDsparkStageCount);
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    EXPECT_EQ(backend.commands[index],
              (DeepSeekDsparkMtpOperatorCommand{
                  deepseek_dspark_stage_id(index).value(),
                  DeepSeekDsparkMtpOperatorKind::kAttention}));
  }
}
TEST(DeepSeekDsparkMtpBlockExecutorTest,
     RejectsVerifyAndOutOfRangeStageBeforeBackendLaunch) {
  Backend backend;
  auto executor = DeepSeekDsparkMtpBlockExecutor::Create(backend).value();
  EXPECT_FALSE(executor.launch(static_cast<DeepSeekDsparkStageId>(3), plan())
                   .ok());
  EXPECT_FALSE(executor.launch(DeepSeekDsparkStageId::kMtp0,
                               plan(DeepSeekPlanPhase::kVerify)).ok());
  auto empty = plan();
  empty.token_count = 0;
  EXPECT_FALSE(executor.launch(DeepSeekDsparkStageId::kMtp0, empty).ok());
  empty = plan();
  empty.sequence_count = 0;
  EXPECT_FALSE(executor.launch(DeepSeekDsparkStageId::kMtp0, empty).ok());
  EXPECT_TRUE(backend.commands.empty());
}
}}  // namespace pih::<anonymous>
