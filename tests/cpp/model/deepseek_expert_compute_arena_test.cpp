#include "pih/model/deepseek_expert_compute_arena.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(DeepSeekExpertComputeArenaTest, PlansTypedCompatibilityScratch) {
  auto layout = DeepSeekExpertComputeArenaLayout::Create(128);
  ASSERT_TRUE(layout.ok());
  auto arena = layout->bind(0x20000000U, layout->required_bytes());
  ASSERT_TRUE(arena.ok());
  EXPECT_EQ(arena->route_input_bf16.bytes, 128U * 4096U * 2U);
  EXPECT_EQ(arena->expert_output_bf16.address,
            arena->route_input_bf16.address);
  EXPECT_EQ(arena->activation_e4m3.bytes, 128U * 4096U);
  EXPECT_EQ(arena->activation_scale_bits.bytes, 128U * 32U);
  EXPECT_EQ(arena->gate_or_middle_bf16.bytes, 128U * 2048U * 2U);
  EXPECT_EQ(arena->up_bf16.bytes, 128U * 2048U * 2U);
  EXPECT_EQ(arena->route_weights_f32.bytes, 128U * 4U);
  EXPECT_EQ(arena->token_indices_u32.bytes, 128U * 4U);
  EXPECT_EQ(arena->error_flag_u32.bytes, 256U);
}

TEST(DeepSeekExpertComputeArenaTest, RequiresStableAlignedBacking) {
  auto layout = DeepSeekExpertComputeArenaLayout::Create(1);
  ASSERT_TRUE(layout.ok());
  EXPECT_EQ(layout->required_bytes(), 21504U);
  EXPECT_FALSE(layout->bind(0, layout->required_bytes()).ok());
  EXPECT_FALSE(layout->bind(0x20000001U, layout->required_bytes()).ok());
  EXPECT_FALSE(layout->bind(0x20000000U,
                            layout->required_bytes() - 1U).ok());
}

TEST(DeepSeekExpertComputeArenaTest, RejectsInvalidTokenCapacity) {
  EXPECT_FALSE(DeepSeekExpertComputeArenaLayout::Create(0).ok());
  EXPECT_FALSE(DeepSeekExpertComputeArenaLayout::Create(
                   DeepSeekExpertComputeArenaLayout::kMaximumTokens + 1U)
                   .ok());
}

}  // namespace
}  // namespace pih
