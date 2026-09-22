#include <cstdint>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_observation_identity.h"

namespace pih {

TEST(QwenSemanticObservationIdentityTest, ReservesCopiesFrontierAndEvent) {
  const auto value =
      reserve_qwen_semantic_observation_identity(100, 200, 57).value();
  EXPECT_EQ(value.first_copy_plan_id, 100);
  EXPECT_EQ(value.copy_plan_count, 58);
  EXPECT_EQ(value.frontier_plan_id, 158);
  EXPECT_EQ(value.next_plan_id, 159);
  EXPECT_EQ(value.completion_event_generation, 200);
  EXPECT_EQ(value.next_event_generation, 201);
}

TEST(QwenSemanticObservationIdentityTest, RejectsZeroAndEveryOverflowEdge) {
  EXPECT_FALSE(reserve_qwen_semantic_observation_identity(0, 1, 1).ok());
  EXPECT_FALSE(reserve_qwen_semantic_observation_identity(1, 0, 1).ok());
  EXPECT_FALSE(reserve_qwen_semantic_observation_identity(1, 1, 0).ok());
  EXPECT_FALSE(reserve_qwen_semantic_observation_identity(
                   UINT64_MAX - 1, 1, 1)
                   .ok());
  EXPECT_FALSE(reserve_qwen_semantic_observation_identity(
                   1, UINT64_MAX, 1)
                   .ok());
}

}  // namespace pih
