#include "pih/model/deepseek_attention_page_arena.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekAttentionPageArenaTest, ResolvesEachTypedPoolWithoutAliasing) {
  auto arena = DeepSeekAttentionPageArena::Create(
      {0x100000, 3U * 65536U}, {0x200000, 3U * 16384U},
      {0x300000, 2U * 65536U}, 3, 2);
  ASSERT_TRUE(arena.ok());
  auto main4 = DeepSeekBlockHandle::Create(
      DeepSeekStatePoolKind::kRatio4MainBf16, 2, 7).value();
  auto index4 = DeepSeekBlockHandle::Create(
      DeepSeekStatePoolKind::kRatio4IndexBf16, 2, 7).value();
  auto main128 = DeepSeekBlockHandle::Create(
      DeepSeekStatePoolKind::kRatio128MainBf16, 1, 9).value();
  EXPECT_EQ(arena->Resolve(main4).value().address, 0x120000U);
  EXPECT_EQ(arena->Resolve(main4).value().bytes, 65536U);
  EXPECT_EQ(arena->Resolve(index4).value().address, 0x208000U);
  EXPECT_EQ(arena->Resolve(index4).value().bytes, 16384U);
  EXPECT_EQ(arena->Resolve(main128).value().address, 0x310000U);
  EXPECT_EQ(arena->Resolve(main128).value().bytes, 65536U);
}

TEST(DeepSeekAttentionPageArenaTest, RejectsWrongGeometryAndForeignHandles) {
  EXPECT_FALSE(DeepSeekAttentionPageArena::Create(
                   {0x100000, 65535}, {0x200000, 16384},
                   {0x300000, 65536}, 1, 1)
                   .ok());
  EXPECT_FALSE(DeepSeekAttentionPageArena::Create(
                   {0x100001, 65536}, {0x200000, 16384},
                   {0x300000, 65536}, 1, 1)
                   .ok());
  auto arena = DeepSeekAttentionPageArena::Create(
      {0x100000, 65536}, {0x200000, 16384}, {0x300000, 65536}, 1, 1)
                   .value();
  auto out_of_range = DeepSeekBlockHandle::Create(
      DeepSeekStatePoolKind::kRatio4MainBf16, 1, 1).value();
  auto recent = DeepSeekBlockHandle::Create(
      DeepSeekStatePoolKind::kRecentBf16, 0, 1).value();
  EXPECT_FALSE(arena.Resolve(out_of_range).ok());
  EXPECT_FALSE(arena.Resolve(recent).ok());
}

} }  // namespace pih
