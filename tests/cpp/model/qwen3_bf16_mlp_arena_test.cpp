#include "pih/model/qwen3_bf16_mlp_arena.h"

#include <cstdint>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_linear_shape.h"

namespace pih {
namespace {

TEST(QwenBf16MlpArenaLayoutTest, UsesTwoAlignedCallerOwnedIntermediates) {
  auto layout = QwenBf16MlpArenaLayout::Create(17);
  ASSERT_TRUE(layout.ok()) << layout.status().message();
  const std::uint64_t expected = UINT64_C(17) * 3072 * 2;
  EXPECT_EQ(layout->gate_offset_bytes(), 0);
  EXPECT_EQ(layout->intermediate_bytes(), expected);
  EXPECT_EQ(layout->up_offset_bytes() % 256, 0);
  EXPECT_GE(layout->up_offset_bytes(), expected);
  EXPECT_GE(layout->arena_bytes(), layout->up_offset_bytes() + expected);
  EXPECT_EQ(layout->arena_bytes() % 256, 0);
}

TEST(QwenBf16MlpArenaLayoutTest, HasExactBoundedMaximumAndRejectsDrift) {
  auto maximum = QwenBf16MlpArenaLayout::Create(
      QwenBf16LinearShape::kMaximumTokensPerPlan);
  ASSERT_TRUE(maximum.ok());
  EXPECT_EQ(maximum->intermediate_bytes(), UINT64_C(4096) * 3072 * 2);
  EXPECT_EQ(maximum->arena_bytes(), UINT64_C(2) * 4096 * 3072 * 2);
  EXPECT_FALSE(QwenBf16MlpArenaLayout::Create(0).ok());
  EXPECT_FALSE(QwenBf16MlpArenaLayout::Create(
                   QwenBf16LinearShape::kMaximumTokensPerPlan + 1)
                   .ok());
}

}  // namespace
}  // namespace pih
