#include "pih/model/qwen3_bf16_execution_arena.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenBf16ExecutionArenaLayoutTest, SeparatesTypedPreallocatedOwners) {
  auto layout = QwenBf16ExecutionArenaLayout::Create(17, 1);
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  const std::array spans{layout->hidden(), layout->normalized(),
                         layout->query(), layout->key(), layout->value(),
                         layout->attention()};
  std::uint64_t previous_end = 0;
  for (const auto span : spans) {
    EXPECT_EQ(span.offset_bytes % 256, 0);
    EXPECT_GE(span.offset_bytes, previous_end);
    previous_end = span.offset_bytes + span.size_bytes;
  }
  EXPECT_EQ(previous_end, layout->activation_arena_bytes());
  EXPECT_EQ(layout->hidden().size_bytes, UINT64_C(17) * 1024 * 2);
  EXPECT_EQ(layout->query().size_bytes, UINT64_C(17) * 2048 * 2);
  EXPECT_EQ(layout->activation_arena_bytes() % 256, 0);
  EXPECT_EQ(layout->mlp().arena_bytes(), UINT64_C(2) * 17 * 3072 * 2);
  EXPECT_EQ(layout->rope_workspace_bytes(), UINT64_C(17) * 512);
  EXPECT_EQ(layout->logit_workspace_bytes(), UINT64_C(151936) * 4);
}

TEST(QwenBf16ExecutionArenaLayoutTest, BoundsMaximumWithoutTokenLogitMatrix) {
  auto layout = QwenBf16ExecutionArenaLayout::Create(4096, 1);
  ASSERT_TRUE(layout.ok());
  EXPECT_EQ(layout->activation_arena_bytes(), UINT64_C(4096) * 8192 * 2);
  EXPECT_EQ(layout->mlp().arena_bytes(), UINT64_C(4096) * 6144 * 2);
  EXPECT_EQ(layout->rope_workspace_bytes(), UINT64_C(4096) * 512);
  EXPECT_EQ(layout->logit_workspace_bytes(), UINT64_C(607744));
  EXPECT_LT(layout->logit_workspace_bytes(), UINT64_C(4096) * 607744);
}

TEST(QwenBf16ExecutionArenaLayoutTest, SizesPackedSampleRowsNotPackedTokens) {
  auto layout = QwenBf16ExecutionArenaLayout::Create(64, 7);
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  EXPECT_EQ(layout->active_logit_rows(), 7);
  EXPECT_EQ(layout->logit_workspace_bytes(), UINT64_C(7) * 151936 * 4);
  EXPECT_LT(layout->logit_workspace_bytes(), UINT64_C(64) * 151936 * 4);
}

TEST(QwenBf16ExecutionArenaLayoutTest, RejectsInvalidTokensAndLogitRows) {
  EXPECT_FALSE(QwenBf16ExecutionArenaLayout::Create(0, 1).ok());
  EXPECT_FALSE(QwenBf16ExecutionArenaLayout::Create(4097, 1).ok());
  EXPECT_FALSE(QwenBf16ExecutionArenaLayout::Create(1, 0).ok());
  EXPECT_FALSE(QwenBf16ExecutionArenaLayout::Create(1, 2).ok());
  EXPECT_FALSE(QwenBf16ExecutionArenaLayout::Create(8, 9).ok());
}

}  // namespace
}  // namespace pih
