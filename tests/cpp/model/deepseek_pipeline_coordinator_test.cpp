#include "pih/model/deepseek_pipeline_coordinator.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

struct CoordinatorFixture final {
  explicit CoordinatorFixture(std::uint32_t world_size) {
    auto capacity = DeepSeekPipelineCapacity::Create(world_size, 8, 8, 1, false);
    EXPECT_TRUE(capacity.ok());
    auto created = DeepSeekPipelineResourceSet::Create(*capacity);
    EXPECT_TRUE(created.ok());
    resources = std::move(*created);
    transaction = resources.prepare(
        {3, 1, DeepSeekPlanPhase::kDecode, 2, 2});
    EXPECT_TRUE(transaction.ok());
  }
  DeepSeekPipelineResourceSet resources;
  Result<DeepSeekPipelineTransaction> transaction =
      Status::Internal("fixture not initialized");
};

TEST(DeepSeekPipelineCoordinatorTest, CommitsOnlyAfterEveryRankReady) {
  CoordinatorFixture fixture(3);
  auto coordinator = DeepSeekPipelineCoordinator::Create(*fixture.transaction, 3);
  ASSERT_TRUE(coordinator.ok());
  ASSERT_TRUE(coordinator->stage_ready(0).ok());
  EXPECT_FALSE(coordinator->commit().ok());
  ASSERT_TRUE(coordinator->stage_ready(2).ok());
  ASSERT_TRUE(coordinator->stage_ready(1).ok());
  EXPECT_EQ(coordinator->state(),
            DeepSeekPipelineCoordinatorState::kReadyToCommit);
  ASSERT_TRUE(coordinator->commit().ok());
  EXPECT_EQ(fixture.transaction->state(),
            DeepSeekPipelineTransactionState::kCommitted);
}

TEST(DeepSeekPipelineCoordinatorTest, RejectAbortsEveryRankReservation) {
  CoordinatorFixture fixture(2);
  auto coordinator = DeepSeekPipelineCoordinator::Create(*fixture.transaction, 2);
  ASSERT_TRUE(coordinator.ok());
  ASSERT_TRUE(coordinator->stage_ready(0).ok());
  EXPECT_FALSE(coordinator->stage_reject(
      1, Status::ResourceExhausted("rank one lacks workspace")).ok());
  EXPECT_EQ(coordinator->state(), DeepSeekPipelineCoordinatorState::kAborted);
  EXPECT_EQ(fixture.transaction->state(),
            DeepSeekPipelineTransactionState::kAborted);
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 2U);
  EXPECT_EQ(fixture.resources.rank(1).control_available(), 2U);
}

TEST(DeepSeekPipelineCoordinatorTest, CancelAfterCommitDrainsBeforeRelease) {
  CoordinatorFixture fixture(2);
  auto coordinator = DeepSeekPipelineCoordinator::Create(*fixture.transaction, 2);
  ASSERT_TRUE(coordinator.ok());
  ASSERT_TRUE(coordinator->stage_ready(0).ok());
  ASSERT_TRUE(coordinator->stage_ready(1).ok());
  ASSERT_TRUE(coordinator->commit().ok());
  ASSERT_TRUE(coordinator->cancel_after_commit().ok());
  EXPECT_TRUE(coordinator->suppress_output());
  ASSERT_TRUE(coordinator->stage_complete(1).ok());
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 1U);
  ASSERT_TRUE(coordinator->stage_complete(0).ok());
  EXPECT_EQ(coordinator->state(), DeepSeekPipelineCoordinatorState::kComplete);
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 2U);
  EXPECT_EQ(fixture.resources.rank(1).control_available(), 2U);
}

TEST(DeepSeekPipelineCoordinatorTest, PostCommitFailurePoisonsWithoutRelease) {
  CoordinatorFixture fixture(1);
  auto coordinator = DeepSeekPipelineCoordinator::Create(*fixture.transaction, 1);
  ASSERT_TRUE(coordinator.ok());
  ASSERT_TRUE(coordinator->stage_ready(0).ok());
  ASSERT_TRUE(coordinator->commit().ok());
  EXPECT_FALSE(coordinator->stage_failed(
      0, Status::Internal("CUDA launch failed")).ok());
  EXPECT_EQ(coordinator->state(), DeepSeekPipelineCoordinatorState::kPoisoned);
  EXPECT_TRUE(coordinator->suppress_output());
  EXPECT_EQ(fixture.resources.rank(0).control_available(), 1U);
}

TEST(DeepSeekPipelineCoordinatorTest, RejectsDuplicateRankSignals) {
  CoordinatorFixture fixture(1);
  auto coordinator = DeepSeekPipelineCoordinator::Create(*fixture.transaction, 1);
  ASSERT_TRUE(coordinator.ok());
  ASSERT_TRUE(coordinator->stage_ready(0).ok());
  EXPECT_FALSE(coordinator->stage_ready(0).ok());
  ASSERT_TRUE(coordinator->commit().ok());
  ASSERT_TRUE(coordinator->stage_complete(0).ok());
  EXPECT_FALSE(coordinator->stage_complete(0).ok());
}

}  // namespace
}  // namespace pih
