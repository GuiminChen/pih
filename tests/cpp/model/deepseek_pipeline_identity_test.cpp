#include "pih/model/deepseek_pipeline_identity.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <utility>

namespace pih {
namespace {

TEST(DeepSeekPipelineIdentityTest,
     PlanRootsAreStableAndDistinctForOneToFourRanks) {
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    auto first = DeepSeekPipelinePlan::Create(world_size, false);
    auto replay = DeepSeekPipelinePlan::Create(world_size, false);
    auto enabled = DeepSeekPipelinePlan::Create(world_size, true);
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(replay.ok());
    ASSERT_TRUE(enabled.ok());
    auto first_root = compile_deepseek_pipeline_plan_root(*first);
    auto replay_root = compile_deepseek_pipeline_plan_root(*replay);
    auto enabled_root = compile_deepseek_pipeline_plan_root(*enabled);
    ASSERT_TRUE(first_root.ok());
    ASSERT_TRUE(replay_root.ok());
    ASSERT_TRUE(enabled_root.ok());
    EXPECT_EQ(*first_root, *replay_root);
    EXPECT_NE(*first_root, *enabled_root);
    if (world_size > 1) {
      auto smaller = DeepSeekPipelinePlan::Create(world_size - 1, false);
      ASSERT_TRUE(smaller.ok());
      auto smaller_root = compile_deepseek_pipeline_plan_root(*smaller);
      ASSERT_TRUE(smaller_root.ok());
      EXPECT_NE(*first_root, *smaller_root);
    }
  }
}

TEST(DeepSeekPipelineIdentityTest, MovedFromPlanFailsClosed) {
  auto plan = DeepSeekPipelinePlan::Create(2, true);
  ASSERT_TRUE(plan.ok());
  auto owner = std::move(*plan);
  EXPECT_TRUE(compile_deepseek_pipeline_plan_root(owner).ok());
  EXPECT_FALSE(compile_deepseek_pipeline_plan_root(*plan).ok());
}

TEST(DeepSeekPipelineIdentityTest,
     CapacityRootsBindEveryRequestedAndDerivedCeiling) {
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    auto first = DeepSeekPipelineCapacity::Create(world_size, 17, 9, 3, true);
    auto replay = DeepSeekPipelineCapacity::Create(world_size, 17, 9, 3, true);
    auto disabled =
        DeepSeekPipelineCapacity::Create(world_size, 17, 9, 3, false);
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(replay.ok());
    ASSERT_TRUE(disabled.ok());
    auto first_root = compile_deepseek_pipeline_capacity_root(*first);
    auto replay_root = compile_deepseek_pipeline_capacity_root(*replay);
    auto disabled_root = compile_deepseek_pipeline_capacity_root(*disabled);
    ASSERT_TRUE(first_root.ok());
    ASSERT_TRUE(replay_root.ok());
    ASSERT_TRUE(disabled_root.ok());
    EXPECT_EQ(*first_root, *replay_root);
    EXPECT_NE(*first_root, *disabled_root);
  }
}

TEST(DeepSeekPipelineIdentityTest, CapacityRejectsEveryClassOfDerivedDrift) {
  auto make_capacity = [] {
    return DeepSeekPipelineCapacity::Create(3, 17, 9, 3, true).value();
  };
  {
    auto capacity = make_capacity();
    ++capacity.max_pipeline_tokens;
    EXPECT_FALSE(compile_deepseek_pipeline_capacity_root(capacity).ok());
  }
  {
    auto capacity = make_capacity();
    ++capacity.expert_tokens;
    EXPECT_FALSE(compile_deepseek_pipeline_capacity_root(capacity).ok());
  }
  {
    auto capacity = make_capacity();
    ++capacity.max_sequences;
    EXPECT_FALSE(compile_deepseek_pipeline_capacity_root(capacity).ok());
  }
  {
    auto capacity = make_capacity();
    ++capacity.boundary_slot_bytes;
    EXPECT_FALSE(compile_deepseek_pipeline_capacity_root(capacity).ok());
  }
  {
    auto capacity = make_capacity();
    ++capacity.control_payload_bytes;
    EXPECT_FALSE(compile_deepseek_pipeline_capacity_root(capacity).ok());
  }
  {
    auto capacity = make_capacity();
    ++capacity.expert_workspace_bytes;
    EXPECT_FALSE(compile_deepseek_pipeline_capacity_root(capacity).ok());
  }
  {
    auto capacity = make_capacity();
    ++capacity.ranks[1].recv_bytes;
    EXPECT_FALSE(compile_deepseek_pipeline_capacity_root(capacity).ok());
  }
  {
    auto capacity = make_capacity();
    capacity.ranks.pop_back();
    EXPECT_FALSE(compile_deepseek_pipeline_capacity_root(capacity).ok());
  }
}

TEST(DeepSeekPipelineIdentityTest, InvalidCapacityInputsFailClosed) {
  DeepSeekPipelineCapacity empty;
  EXPECT_FALSE(compile_deepseek_pipeline_capacity_root(empty).ok());

  auto overflow = DeepSeekPipelineCapacity::Create(
      1, 1, 1, std::numeric_limits<std::uint32_t>::max(), true);
  EXPECT_FALSE(overflow.ok());
}

TEST(DeepSeekPipelineIdentityTest, CanonicalAbiHasGoldenRoots) {
  auto plan = DeepSeekPipelinePlan::Create(1, false);
  auto capacity = DeepSeekPipelineCapacity::Create(1, 17, 9, 3, true);
  ASSERT_TRUE(plan.ok());
  ASSERT_TRUE(capacity.ok());
  auto plan_root = compile_deepseek_pipeline_plan_root(*plan);
  auto capacity_root = compile_deepseek_pipeline_capacity_root(*capacity);
  ASSERT_TRUE(plan_root.ok());
  ASSERT_TRUE(capacity_root.ok());
  EXPECT_EQ(plan_root->hex(),
            "19e1f1e704d0a94b71d0f2a7cdadfcbf24b06a82bf287965f3dac4d3d3006f1a");
  EXPECT_EQ(capacity_root->hex(),
            "740ca6de48982e43b5ef3c447063a7a0e351f2bbb3c4c5de8277014cd3382e9c");
}

}  // namespace
}  // namespace pih
