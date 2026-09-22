#include "pih/model/qwen3_bf16_packed_kv_metadata.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest root(std::string_view value) {
  return sha256(std::as_bytes(std::span(value))).value();
}

QwenKvBlockTable table(std::uint32_t owner, std::uint32_t generation,
                       std::uint32_t first_slot) {
  const std::array handles{QwenKvBlockHandle{first_slot, 1},
                           QwenKvBlockHandle{first_slot + 1, 1}};
  auto result = QwenKvBlockTable::Create(owner, generation, 32, handles).value();
  auto initial = result.prepare_append(owner == 0 ? 15 : 16).value();
  EXPECT_TRUE(result.commit_append(initial).ok());
  return result;
}

struct Fixture final {
  PackedTokenPlan plan;
  PackedTokenMetadataArena token_arena;
  PackedTokenMetadataView tokens;
};

Fixture fixture() {
  const std::array<std::uint32_t, 2> first{7, 8};
  const std::array<std::uint32_t, 1> second{9};
  const std::array inputs{
      PackedSequenceInput{11, 15, 2, 1,
                          packed_token_input_digest(first).value()},
      PackedSequenceInput{12, 16, 1, 1,
                          packed_token_input_digest(second).value()}};
  auto plan = PackedTokenPlan::Create(
      7, 1, PackedTokenPhase::kPrefill, "profile-r1", root("resources"),
      inputs, 5, {2, 3, 5}).value();
  auto arena = PackedTokenMetadataArena::Create({2, 5}).value();
  const std::array payloads{PackedSequenceTokens{11, first, false},
                            PackedSequenceTokens{12, second, true}};
  auto tokens = arena.materialize(plan, payloads).value();
  return {std::move(plan), std::move(arena), tokens};
}

TEST(QwenBf16PackedKvMetadataTest,
     FlattensCrossBlockAppendAndVisibleTablesWithDummySentinels) {
  auto data = fixture();
  auto first = table(0, 1, 10);
  auto second = table(1, 2, 20);
  const std::array bindings{
      QwenBf16PackedKvBinding{11, &first},
      QwenBf16PackedKvBinding{12, &second}};
  const std::array appends{first.prepare_append(17).value(),
                           second.prepare_append(17).value()};
  auto arena = QwenBf16PackedKvMetadataArena::Create({2, 5, 4}).value();

  auto view = arena.materialize(data.plan, data.tokens, bindings, appends);
  ASSERT_TRUE(view.ok()) << view.status().message();
  ASSERT_EQ(view->append_handles.size(), 5U);
  EXPECT_EQ(view->append_handles[0], (QwenKvBlockHandle{10, 1}));
  EXPECT_EQ(view->append_handles[1], (QwenKvBlockHandle{11, 1}));
  EXPECT_EQ(view->append_handles[2], (QwenKvBlockHandle{21, 1}));
  EXPECT_EQ(view->append_handles[3], kInvalidPackedKvBlockHandle);
  EXPECT_EQ(view->append_handles[4], kInvalidPackedKvBlockHandle);
  EXPECT_EQ(view->token_offsets[0], 15U);
  EXPECT_EQ(view->token_offsets[1], 0U);
  EXPECT_EQ(view->token_offsets[2], 0U);
  ASSERT_EQ(view->visible_handle_offsets.size(), 3U);
  EXPECT_EQ(view->visible_handle_offsets[0], 0U);
  EXPECT_EQ(view->visible_handle_offsets[1], 2U);
  EXPECT_EQ(view->visible_handle_offsets[2], 4U);
  ASSERT_EQ(view->visible_handles.size(), 4U);
  EXPECT_EQ(view->visible_handles[0], (QwenKvBlockHandle{10, 1}));
  EXPECT_EQ(view->visible_handles[1], (QwenKvBlockHandle{11, 1}));
  EXPECT_EQ(view->visible_handles[2], (QwenKvBlockHandle{20, 1}));
  EXPECT_EQ(view->visible_handles[3], (QwenKvBlockHandle{21, 1}));
  EXPECT_EQ(view->key_token_counts[0], 17U);
  EXPECT_EQ(view->key_token_counts[1], 17U);
  EXPECT_EQ(view->owner_sequence_indices[0], 0U);
  EXPECT_EQ(view->owner_sequence_indices[1], 1U);
}

TEST(QwenBf16PackedKvMetadataTest,
     RejectsStaleAppendWithoutPublishingANewGeneration) {
  auto data = fixture();
  auto first = table(0, 1, 10);
  auto second = table(1, 2, 20);
  const std::array bindings{
      QwenBf16PackedKvBinding{11, &first},
      QwenBf16PackedKvBinding{12, &second}};
  std::array appends{first.prepare_append(17).value(),
                     second.prepare_append(17).value()};
  auto arena = QwenBf16PackedKvMetadataArena::Create({2, 5, 4}).value();
  const auto first_view =
      arena.materialize(data.plan, data.tokens, bindings, appends).value();
  appends[1].expected_block_table_generation++;
  EXPECT_FALSE(arena.materialize(data.plan, data.tokens, bindings, appends).ok());
  appends[1].expected_block_table_generation--;
  const auto second_view =
      arena.materialize(data.plan, data.tokens, bindings, appends).value();
  EXPECT_EQ(second_view.generation, first_view.generation + 1);
  EXPECT_EQ(second_view.append_handles.data(), first_view.append_handles.data());
}

}  // namespace
}  // namespace pih
