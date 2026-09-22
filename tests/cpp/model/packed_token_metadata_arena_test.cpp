#include "pih/model/packed_token_metadata_arena.h"

#include <array>
#include <limits>

#include <gtest/gtest.h>

namespace pih {
namespace {
Sha256Digest root() {
  constexpr std::string_view value = "root";
  return sha256(std::as_bytes(std::span(value))).value();
}

TEST(PackedTokenMetadataArenaTest, MaterializesRealPrefixSamplesAndDummyPadding) {
  const std::array<std::uint32_t, 2> a{4, 5};
  const std::array<std::uint32_t, 1> b{9};
  const std::array inputs{
      PackedSequenceInput{11, 7, 2, 1, packed_token_input_digest(a).value()},
      PackedSequenceInput{12, 20, 1, 2, packed_token_input_digest(b).value()}};
  auto plan = PackedTokenPlan::Create(1, 1, PackedTokenPhase::kPrefill, "p",
      root(), inputs, 5, {2, 3, 5}).value();
  auto arena = PackedTokenMetadataArena::Create({2, 5}).value();
  const std::array payloads{PackedSequenceTokens{11, a, false},
                            PackedSequenceTokens{12, b, true}};
  auto view = arena.materialize(plan, payloads);
  ASSERT_TRUE(view.ok()) << view.status().message();
  ASSERT_EQ(view->input_token_ids.size(), 5U);
  ASSERT_EQ(view->positions.size(), 5U);
  const std::array<std::uint32_t, 5> expected_tokens{4, 5, 9, 0, 0};
  const std::array<std::uint64_t, 5> expected_positions{7, 8, 20, 0, 0};
  for (std::size_t i = 0; i < expected_tokens.size(); ++i) {
    EXPECT_EQ(view->input_token_ids[i], expected_tokens[i]);
    EXPECT_EQ(view->positions[i], expected_positions[i]);
  }
  EXPECT_EQ(view->request_index[3], kInvalidPackedRequestIndex);
  EXPECT_EQ(view->request_index[4], kInvalidPackedRequestIndex);
  ASSERT_EQ(view->query_start_offsets.size(), 3U);
  EXPECT_EQ(view->query_start_offsets[0], 0U);
  EXPECT_EQ(view->query_start_offsets[1], 2U);
  EXPECT_EQ(view->query_start_offsets[2], 3U);
  ASSERT_EQ(view->sample_row_index.size(), 1U);
  EXPECT_EQ(view->sample_row_index[0], 2U);
}

TEST(PackedTokenMetadataArenaTest, RejectsMismatchWithoutPublishingMutation) {
  const std::array<std::uint32_t, 1> tokens{4};
  const std::array inputs{PackedSequenceInput{
      11, 0, 1, 1, packed_token_input_digest(tokens).value()}};
  auto plan = PackedTokenPlan::Create(1, 1, PackedTokenPhase::kDecode, "p",
      root(), inputs, 1, {1, 1, 1}).value();
  auto arena = PackedTokenMetadataArena::Create({1, 1}).value();
  const std::array good{PackedSequenceTokens{11, tokens, true}};
  auto first = arena.materialize(plan, good).value();
  const auto generation = first.generation;
  const auto address = first.input_token_ids.data();
  const std::array<std::uint32_t, 1> wrong{5};
  const std::array bad{PackedSequenceTokens{11, wrong, true}};
  EXPECT_FALSE(arena.materialize(plan, bad).ok());
  auto second = arena.materialize(plan, good).value();
  EXPECT_EQ(second.generation, generation + 1);
  EXPECT_EQ(second.input_token_ids.data(), address);
}

TEST(PackedTokenMetadataArenaTest, RejectsPositionOverflowAndCapacityMismatch) {
  const std::array<std::uint32_t, 2> tokens{1, 2};
  const std::array inputs{PackedSequenceInput{
      1, std::numeric_limits<std::uint64_t>::max(), 2, 1,
      packed_token_input_digest(tokens).value()}};
  auto plan = PackedTokenPlan::Create(1, 1, PackedTokenPhase::kPrefill, "p",
      root(), inputs, 2, {1, 2, 2}).value();
  auto arena = PackedTokenMetadataArena::Create({1, 2}).value();
  const std::array payload{PackedSequenceTokens{1, tokens, false}};
  EXPECT_FALSE(arena.materialize(plan, payload).ok());
  auto small = PackedTokenMetadataArena::Create({1, 1}).value();
  EXPECT_FALSE(small.materialize(plan, payload).ok());
}
}  // namespace
}  // namespace pih
