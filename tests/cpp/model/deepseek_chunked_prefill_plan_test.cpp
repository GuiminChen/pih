#include "pih/model/deepseek_chunked_prefill_plan.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekChunkedPrefillPlanTest, Splits513TokensAt512Boundary) {
  const std::array<DeepSeekChunkedPrefillSequence, 1> first_input{{
      {11, 7, 513, 0},
  }};
  auto first = DeepSeekChunkedPrefillPlan::Compile(1, 512, first_input);
  ASSERT_TRUE(first.ok()) << first.status().message();
  ASSERT_EQ(first->slices().size(), 1U);
  EXPECT_EQ(first->slices()[0].start_position, 0U);
  EXPECT_EQ(first->slices()[0].token_count, 512U);
  EXPECT_EQ(first->token_count(), 512U);
  EXPECT_FALSE(first->publishes_sampled_token());

  const std::array<DeepSeekChunkedPrefillSequence, 1> second_input{{
      {11, 7, 513, 512},
  }};
  auto second = DeepSeekChunkedPrefillPlan::Compile(2, 512, second_input);
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(second->slices()[0].start_position, 512U);
  EXPECT_EQ(second->slices()[0].token_count, 1U);
  EXPECT_TRUE(second->publishes_sampled_token());
  EXPECT_NE(first->identity(), second->identity());
}

TEST(DeepSeekChunkedPrefillPlanTest, PreservesStableSequenceOrder) {
  const std::array<DeepSeekChunkedPrefillSequence, 2> input{{
      {11, 7, 6, 0}, {12, 8, 6, 0},
  }};
  auto plan = DeepSeekChunkedPrefillPlan::Compile(1, 8, input);
  ASSERT_TRUE(plan.ok());
  ASSERT_EQ(plan->slices().size(), 2U);
  EXPECT_EQ(plan->slices()[0].sequence_index, 0U);
  EXPECT_EQ(plan->slices()[0].token_count, 6U);
  EXPECT_EQ(plan->slices()[1].sequence_index, 1U);
  EXPECT_EQ(plan->slices()[1].token_count, 2U);
  EXPECT_FALSE(plan->publishes_sampled_token());
}

TEST(DeepSeekChunkedPrefillPlanTest, RejectsInvalidOrCompletedInput) {
  EXPECT_EQ(DeepSeekChunkedPrefillPlan::Compile(0, 8, {}).status().code(),
            StatusCode::kInvalidArgument);
  const std::array<DeepSeekChunkedPrefillSequence, 1> completed{{
      {11, 7, 6, 6},
  }};
  EXPECT_EQ(DeepSeekChunkedPrefillPlan::Compile(1, 8, completed).status().code(),
            StatusCode::kFailedPrecondition);
  const std::array<DeepSeekChunkedPrefillSequence, 2> duplicated{{
      {11, 7, 6, 0}, {11, 8, 6, 0},
  }};
  EXPECT_EQ(DeepSeekChunkedPrefillPlan::Compile(1, 8, duplicated).status().code(),
            StatusCode::kInvalidArgument);
}

} }  // namespace pih
