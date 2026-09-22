#include "pih/model/qwen3_bf16_step_input_plan.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

QwenKvBlockTable table(std::uint32_t reserved_tokens = 33) {
  const QwenKvBlockHandle handles[] = {{4, 9}, {7, 3}, {8, 5}};
  return QwenKvBlockTable::Create(2, 6, reserved_tokens, handles).value();
}

TEST(QwenBf16StepInputPlanTest, MaterializesCrossBlockPrefillInputs) {
  auto blocks = table();
  auto append = blocks.prepare_append(17);
  ASSERT_TRUE(append.ok());
  std::vector<std::int64_t> tokens(17);
  for (std::int64_t index = 0; index < 17; ++index) tokens[index] = index + 10;
  auto plan = QwenBf16StepInputPlan::Create(tokens, 0, blocks, *append);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->positions().front(), 0);
  EXPECT_EQ(plan->positions().back(), 16);
  EXPECT_EQ(plan->append_handles().front(), (QwenKvBlockHandle{4, 9}));
  EXPECT_EQ(plan->append_handles().back(), (QwenKvBlockHandle{7, 3}));
  EXPECT_EQ(plan->token_offsets().front(), 0);
  EXPECT_EQ(plan->token_offsets()[15], 15);
  EXPECT_EQ(plan->token_offsets().back(), 0);
  EXPECT_EQ(plan->visible_handles().size(), 2);
  EXPECT_EQ(plan->key_token_count(), 17);
}

TEST(QwenBf16StepInputPlanTest, MaterializesSingleDecodeAtCommittedFrontier) {
  auto blocks = table();
  auto first = blocks.prepare_append(16).value();
  ASSERT_TRUE(blocks.commit_append(first).ok());
  auto append = blocks.prepare_append(17).value();
  const std::int64_t token[] = {42};
  auto plan = QwenBf16StepInputPlan::Create(token, 16, blocks, append);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan->positions()[0], 16);
  EXPECT_EQ(plan->append_handles()[0], (QwenKvBlockHandle{7, 3}));
  EXPECT_EQ(plan->token_offsets()[0], 0);
  EXPECT_EQ(plan->visible_handles().size(), 2);
}

TEST(QwenBf16StepInputPlanTest, RejectsTransactionAndTokenDriftAtomically) {
  auto blocks = table();
  auto append = blocks.prepare_append(1).value();
  const std::int64_t token[] = {42};
  EXPECT_FALSE(QwenBf16StepInputPlan::Create(token, 1, blocks, append).ok());
  append.expected_block_table_generation++;
  EXPECT_FALSE(QwenBf16StepInputPlan::Create(token, 0, blocks, append).ok());
  append = blocks.prepare_append(1).value();
  const std::int64_t invalid[] = {151936};
  EXPECT_FALSE(QwenBf16StepInputPlan::Create(invalid, 0, blocks, append).ok());
}

}  // namespace
}  // namespace pih
