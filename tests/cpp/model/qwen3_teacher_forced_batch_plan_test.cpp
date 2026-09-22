#include <algorithm>
#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_batch_plan.h"

namespace pih {
namespace {

TEST(QwenTeacherForcedBatchPlanTest,
     MapsMultipleTargetsPerSequenceToPackedRows) {
  const QwenTeacherForcedChunk chunk{
      QwenTeacherForcedCategory::kCode, 64, 4,
      4ULL * QwenTeacherForcedChunkPlan::kVocabularySize * sizeof(float)};
  const std::array<std::uint32_t, 3> sequence_tokens{3, 5, 2};
  const std::array<QwenTeacherForcedTarget, 4> targets{{
      {1, 0, 17}, {0, 2, 23}, {1, 4, 42}, {2, 1, 151935},
  }};

  auto plan = QwenTeacherForcedBatchPlan::Create(
      chunk, sequence_tokens, targets);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->execution_tokens(), 10U);
  EXPECT_EQ(plan->sequence_count(), 3U);
  EXPECT_EQ(plan->sample_count(), 4U);
  EXPECT_EQ(plan->chunk().category, QwenTeacherForcedCategory::kCode);
  EXPECT_EQ(plan->chunk().category_row_begin, 64U);
  EXPECT_EQ(plan->chunk().rows, 4U);
  const std::array<std::uint32_t, 4> expected_rows{3, 2, 7, 9};
  const std::array<std::uint32_t, 4> expected_tokens{17, 23, 42, 151935};
  EXPECT_TRUE(std::ranges::equal(plan->sample_row_indices(), expected_rows));
  EXPECT_TRUE(std::ranges::equal(plan->target_token_ids(), expected_tokens));
}

TEST(QwenTeacherForcedBatchPlanTest, RejectsChunkOrCausalTargetDrift) {
  const QwenTeacherForcedChunk chunk{
      QwenTeacherForcedCategory::kEn, 0, 2,
      2ULL * QwenTeacherForcedChunkPlan::kVocabularySize * sizeof(float)};
  const std::array<std::uint32_t, 1> sequence_tokens{4};
  const std::array<QwenTeacherForcedTarget, 2> valid{{
      {0, 0, 1}, {0, 3, 2},
  }};
  auto short_targets = std::span<const QwenTeacherForcedTarget>(valid).first(1);
  EXPECT_FALSE(QwenTeacherForcedBatchPlan::Create(
                   chunk, sequence_tokens, short_targets).ok());

  auto bad_row = valid;
  bad_row[1].token_row = 4;
  EXPECT_FALSE(QwenTeacherForcedBatchPlan::Create(
                   chunk, sequence_tokens, bad_row).ok());
  auto bad_token = valid;
  bad_token[1].target_token_id = 151936;
  EXPECT_FALSE(QwenTeacherForcedBatchPlan::Create(
                   chunk, sequence_tokens, bad_token).ok());
}

TEST(QwenTeacherForcedBatchPlanTest, RejectsPackedCapacityOverflow) {
  const QwenTeacherForcedChunk chunk{
      QwenTeacherForcedCategory::kZh, 0, 1,
      QwenTeacherForcedChunkPlan::kVocabularySize * sizeof(float)};
  const std::array<std::uint32_t, 2> sequence_tokens{4096, 1};
  const std::array<QwenTeacherForcedTarget, 1> targets{{{0, 0, 1}}};
  EXPECT_FALSE(QwenTeacherForcedBatchPlan::Create(
                   chunk, sequence_tokens, targets).ok());
}

TEST(QwenTeacherForcedBatchPlanTest, MaterializesBothPinnedPrefixesAtomically) {
  const QwenTeacherForcedChunk chunk{
      QwenTeacherForcedCategory::kTool, 0, 2,
      2ULL * QwenTeacherForcedChunkPlan::kVocabularySize * sizeof(float)};
  const std::array<std::uint32_t, 1> sequence_tokens{5};
  const std::array<QwenTeacherForcedTarget, 2> targets{{
      {0, 1, 7}, {0, 4, 11},
  }};
  auto plan = QwenTeacherForcedBatchPlan::Create(
      chunk, sequence_tokens, targets).value();
  std::array<std::byte, 16> rows{};
  std::array<std::byte, 16> ids{};
  ASSERT_TRUE(plan.materialize_inputs(rows, ids).ok());
  std::array<std::uint32_t, 2> observed_rows{};
  std::array<std::uint32_t, 2> observed_ids{};
  std::memcpy(observed_rows.data(), rows.data(), sizeof(observed_rows));
  std::memcpy(observed_ids.data(), ids.data(), sizeof(observed_ids));
  EXPECT_EQ(observed_rows, (std::array<std::uint32_t, 2>{1, 4}));
  EXPECT_EQ(observed_ids, (std::array<std::uint32_t, 2>{7, 11}));

  rows.fill(std::byte{0x5a});
  ids.fill(std::byte{0x6b});
  EXPECT_FALSE(plan.materialize_inputs(
      rows, std::span<std::byte>(ids).first(7)).ok());
  EXPECT_TRUE(std::ranges::all_of(rows,
      [](std::byte value) { return value == std::byte{0x5a}; }));
  EXPECT_TRUE(std::ranges::all_of(ids,
      [](std::byte value) { return value == std::byte{0x6b}; }));
}

}  // namespace
}  // namespace pih
