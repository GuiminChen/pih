#include "pih/model/deepseek_pipeline_execution.h"

#include <gtest/gtest.h>

namespace pih { namespace {

struct ExecutionFixture final {
  ExecutionFixture() {
    registry.submit(10, 1);
    registry.submit(11, 1);
    auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
    resources = DeepSeekPipelineResourceSet::Create(*capacity).value();
  }
  DeepSeekRequestRegistry registry;
  DeepSeekPipelineResourceSet resources;
};

TEST(DeepSeekPipelineExecutionTest, MixedCancelAfterCommitDrainsWholePlan) {
  ExecutionFixture fixture;
  auto execution = DeepSeekPipelineExecution::Create(
      fixture.registry, fixture.resources,
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 2}, {{10, 1}, {11, 1}});
  ASSERT_TRUE(execution.ok()) << execution.status().message();
  ASSERT_TRUE(execution->stage_ready(0).ok());
  ASSERT_TRUE(execution->stage_ready(1).ok());
  ASSERT_TRUE(execution->commit().ok());
  ASSERT_TRUE(execution->cancel_request(10, 1).ok());
  EXPECT_EQ(fixture.registry.state(10, 1).value(),
            DeepSeekRequestState::kDraining);
  ASSERT_TRUE(execution->stage_complete(1).ok());
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 1U);
  ASSERT_TRUE(execution->stage_complete(0).ok());
  EXPECT_EQ(fixture.registry.state(10, 1).value(),
            DeepSeekRequestState::kCancelled);
  EXPECT_EQ(fixture.registry.state(11, 1).value(),
            DeepSeekRequestState::kCompleted);
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 2U);
}

TEST(DeepSeekPipelineExecutionTest, CancelBeforeCommitAbortsWholePlan) {
  ExecutionFixture fixture;
  auto execution = DeepSeekPipelineExecution::Create(
      fixture.registry, fixture.resources,
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 2}, {{10, 1}, {11, 1}});
  ASSERT_TRUE(execution.ok());
  ASSERT_TRUE(execution->stage_ready(0).ok());
  EXPECT_TRUE(execution->validate_stage_reject(1).ok());
  EXPECT_TRUE(execution->validate_cancel_request(10, 1).ok());
  EXPECT_EQ(execution->state(), DeepSeekPipelineCoordinatorState::kPreparing);
  EXPECT_EQ(fixture.registry.state(10, 1).value(),
            DeepSeekRequestState::kPrepared);
  ASSERT_TRUE(execution->cancel_request(10, 1).ok());
  EXPECT_EQ(execution->state(), DeepSeekPipelineCoordinatorState::kAborted);
  EXPECT_EQ(fixture.registry.state(10, 1).value(),
            DeepSeekRequestState::kCancelled);
  EXPECT_EQ(fixture.registry.state(11, 1).value(),
            DeepSeekRequestState::kAdmitted);
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 2U);
}

TEST(DeepSeekPipelineExecutionTest, PreparePreflightDoesNotReserveAnyOwner) {
  ExecutionFixture fixture;
  ASSERT_TRUE(fixture.registry.cancel(11, 1).ok());
  auto execution = DeepSeekPipelineExecution::Create(
      fixture.registry, fixture.resources,
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 2}, {{10, 1}, {11, 1}});
  EXPECT_FALSE(execution.ok());
  EXPECT_EQ(fixture.registry.state(10, 1).value(),
            DeepSeekRequestState::kAdmitted);
  EXPECT_EQ(fixture.registry.state(11, 1).value(),
            DeepSeekRequestState::kCancelled);
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 2U);
  EXPECT_EQ(fixture.resources.rank(1).control_available(), 2U);
}

TEST(DeepSeekPipelineExecutionTest, CommitPreflightLeavesAllOwnersPrepared) {
  ExecutionFixture fixture;
  auto execution = DeepSeekPipelineExecution::Create(
      fixture.registry, fixture.resources,
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 2}, {{10, 1}, {11, 1}});
  ASSERT_TRUE(execution.ok());
  EXPECT_FALSE(execution->validate_commit().ok());
  ASSERT_TRUE(execution->stage_ready(0).ok());
  ASSERT_TRUE(execution->stage_ready(1).ok());
  EXPECT_TRUE(execution->validate_commit().ok());
  EXPECT_EQ(execution->state(),
            DeepSeekPipelineCoordinatorState::kReadyToCommit);
  EXPECT_EQ(fixture.registry.state(10, 1).value(),
            DeepSeekRequestState::kPrepared);
  EXPECT_EQ(fixture.registry.state(11, 1).value(),
            DeepSeekRequestState::kPrepared);
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 1U);
  ASSERT_TRUE(execution->commit().ok());
}

TEST(DeepSeekPipelineExecutionTest, StageFailureMarksRequestsFailedDraining) {
  ExecutionFixture fixture;
  auto execution = DeepSeekPipelineExecution::Create(
      fixture.registry, fixture.resources,
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 2}, {{10, 1}, {11, 1}});
  ASSERT_TRUE(execution.ok());
  ASSERT_TRUE(execution->stage_ready(0).ok());
  ASSERT_TRUE(execution->stage_ready(1).ok());
  ASSERT_TRUE(execution->commit().ok());
  EXPECT_FALSE(execution->stage_failed(
      1, Status::Internal("injected rank failure")).ok());
  EXPECT_EQ(execution->state(), DeepSeekPipelineCoordinatorState::kPoisoned);
  EXPECT_EQ(fixture.registry.state(10, 1).value(),
            DeepSeekRequestState::kDraining);
  EXPECT_EQ(fixture.registry.state(11, 1).value(),
            DeepSeekRequestState::kDraining);
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 1U);
}

TEST(DeepSeekPipelineExecutionTest, InvalidStageFailureDoesNotFailRequests) {
  ExecutionFixture fixture;
  auto execution = DeepSeekPipelineExecution::Create(
      fixture.registry, fixture.resources,
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 2}, {{10, 1}, {11, 1}});
  ASSERT_TRUE(execution.ok());
  ASSERT_TRUE(execution->stage_ready(0).ok());
  ASSERT_TRUE(execution->stage_ready(1).ok());
  ASSERT_TRUE(execution->commit().ok());
  EXPECT_FALSE(execution->validate_stage_failed(2).ok());
  EXPECT_FALSE(execution->stage_failed(
      2, Status::Internal("forged rank failure")).ok());
  EXPECT_EQ(execution->state(), DeepSeekPipelineCoordinatorState::kCommitted);
  EXPECT_EQ(fixture.registry.state(10, 1).value(),
            DeepSeekRequestState::kCommitted);
  EXPECT_EQ(fixture.registry.state(11, 1).value(),
            DeepSeekRequestState::kCommitted);
}

TEST(DeepSeekPipelineExecutionTest, CompletionPreflightRetainsPipelineOwnership) {
  ExecutionFixture fixture;
  auto execution = DeepSeekPipelineExecution::Create(
      fixture.registry, fixture.resources,
      {3, 1, DeepSeekPlanPhase::kDecode, 2, 2}, {{10, 1}, {11, 1}});
  ASSERT_TRUE(execution.ok());
  ASSERT_TRUE(execution->stage_ready(0).ok());
  ASSERT_TRUE(execution->stage_ready(1).ok());
  ASSERT_TRUE(execution->commit().ok());
  ASSERT_TRUE(fixture.registry.backend_complete(10, 1, 1).ok());
  EXPECT_FALSE(execution->validate_stage_complete(0).ok());
  EXPECT_FALSE(execution->stage_complete(0).ok());
  EXPECT_EQ(execution->state(), DeepSeekPipelineCoordinatorState::kCommitted);
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 1U);
}

TEST(DeepSeekPipelineExecutionTest, NonTerminalPlanRequeuesRequest) {
  ExecutionFixture fixture;
  auto execution = DeepSeekPipelineExecution::Create(
      fixture.registry, fixture.resources,
      {3, 1, DeepSeekPlanPhase::kPrefill, 2, 1}, {{10, 1, false}});
  ASSERT_TRUE(execution.ok());
  ASSERT_TRUE(execution->stage_ready(0).ok());
  ASSERT_TRUE(execution->stage_ready(1).ok());
  ASSERT_TRUE(execution->commit().ok());
  ASSERT_TRUE(execution->stage_complete(0).ok());
  ASSERT_TRUE(execution->stage_complete(1).ok());
  EXPECT_EQ(fixture.registry.state(10, 1).value(),
            DeepSeekRequestState::kAdmitted);
}

} }  // namespace pih
