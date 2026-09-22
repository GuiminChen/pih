#include <array>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_pair_accumulator.h"

namespace pih {
namespace {

std::array<QwenTeacherForcedChunk, 6> pair_chunks() {
  std::array<QwenTeacherForcedChunk, 6> chunks{};
  for (std::size_t index = 0; index < chunks.size(); ++index) {
    chunks[index] = {static_cast<QwenTeacherForcedCategory>(index), 0, 2,
        2ULL * QwenTeacherForcedChunkPlan::kVocabularySize * sizeof(float)};
  }
  return chunks;
}

QwenTeacherForcedMetricBatch metric_batch(std::uint32_t first,
                                           std::uint32_t second,
                                           double first_nll,
                                           double second_nll) {
  return {{{first, first_nll, true}, {second, second_nll, true}},
          first_nll + second_nll, 0};
}

TEST(QwenTeacherForcedPairAccumulatorTest,
     AccumulatesExactOrderedPairsPerCategory) {
  const auto chunks = pair_chunks();
  auto accumulator = QwenTeacherForcedPairAccumulator::Create(chunks).value();
  for (const auto& chunk : chunks) {
    auto bf16 = metric_batch(1, 2, 0.25, 0.5);
    auto int4 = metric_batch(1, 3, 0.5, 0.75);
    ASSERT_TRUE(accumulator.append(chunk, bf16, int4).ok());
  }
  auto result = accumulator.finalize();
  ASSERT_TRUE(result.ok()) << result.status().message();
  for (std::size_t index = 0; index < result->size(); ++index) {
    EXPECT_EQ((*result)[index].category,
              static_cast<QwenTeacherForcedCategory>(index));
    EXPECT_EQ((*result)[index].evaluated_positions, 2U);
    EXPECT_EQ((*result)[index].equal_argmax_positions, 1U);
    EXPECT_DOUBLE_EQ((*result)[index].bf16_nll_sum, 0.75);
    EXPECT_DOUBLE_EQ((*result)[index].int4_nll_sum, 1.25);
    EXPECT_EQ((*result)[index].nonfinite_count, 0U);
    const auto deltas = (*result)[index].paired_nll_deltas.values();
    ASSERT_EQ(deltas.size(), 2U);
    EXPECT_DOUBLE_EQ(deltas[0], 0.25);
    EXPECT_DOUBLE_EQ(deltas[1], 0.25);
  }
}

TEST(QwenTeacherForcedPairAccumulatorTest,
     RejectsSkippedChunkAndIncompleteFinalization) {
  const auto chunks = pair_chunks();
  auto accumulator = QwenTeacherForcedPairAccumulator::Create(chunks).value();
  auto batch = metric_batch(1, 2, 0.25, 0.5);
  EXPECT_FALSE(accumulator.append(chunks[1], batch, batch).ok());
  EXPECT_FALSE(accumulator.finalize().ok());
}

TEST(QwenTeacherForcedPairAccumulatorTest,
     CountsNonfinitePairsWithoutTrustingBatchSums) {
  const auto chunks = pair_chunks();
  auto accumulator = QwenTeacherForcedPairAccumulator::Create(chunks).value();
  QwenTeacherForcedMetricBatch bf16{{{1, 0.25, true}, {2, 0.0, false}},
                                     99.0, 1};
  QwenTeacherForcedMetricBatch int4{{{1, 0.5, true}, {2, 0.75, true}},
                                     101.0, 0};
  ASSERT_TRUE(accumulator.append(chunks[0], bf16, int4).ok());
  for (std::size_t index = 1; index < chunks.size(); ++index) {
    auto batch = metric_batch(1, 2, 0.0, 0.0);
    ASSERT_TRUE(accumulator.append(chunks[index], batch, batch).ok());
  }
  auto result = accumulator.finalize().value();
  EXPECT_DOUBLE_EQ(result[0].bf16_nll_sum, 0.25);
  EXPECT_DOUBLE_EQ(result[0].int4_nll_sum, 1.25);
  EXPECT_EQ(result[0].nonfinite_count, 1U);
  const auto deltas = result[0].paired_nll_deltas.values();
  ASSERT_EQ(deltas.size(), 1U);
  EXPECT_DOUBLE_EQ(deltas[0], 0.25);
}

TEST(QwenTeacherForcedPairRunCollectorTest,
     StagesOneBf16BatchThenPairsTheMatchingInt4Batch) {
  const auto chunks = pair_chunks();
  auto collector = QwenTeacherForcedPairRunCollector::Create(chunks).value();
  for (const auto& chunk : chunks) {
    ASSERT_TRUE(collector.stage_bf16(
        chunk, metric_batch(1, 2, 0.25, 0.5)).ok());
    ASSERT_TRUE(collector.stage_int4(
        chunk, metric_batch(1, 3, 0.5, 0.75)).ok());
  }
  auto result = collector.finalize();
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ((*result)[0].evaluated_positions, 2U);
  EXPECT_EQ((*result)[0].paired_nll_deltas.values().size(), 2U);
}

TEST(QwenTeacherForcedPairRunCollectorTest,
     RejectsRoleReorderChunkDriftAndUnpairedFinalize) {
  const auto chunks = pair_chunks();
  auto collector = QwenTeacherForcedPairRunCollector::Create(chunks).value();
  auto batch = metric_batch(1, 2, 0.25, 0.5);
  EXPECT_FALSE(collector.stage_int4(chunks[0], batch).ok());

  auto second = QwenTeacherForcedPairRunCollector::Create(chunks).value();
  ASSERT_TRUE(second.stage_bf16(chunks[0], batch).ok());
  EXPECT_FALSE(second.stage_bf16(chunks[0], batch).ok());
  EXPECT_FALSE(second.finalize().ok());

  auto third = QwenTeacherForcedPairRunCollector::Create(chunks).value();
  ASSERT_TRUE(third.stage_bf16(chunks[0], batch).ok());
  EXPECT_FALSE(third.stage_int4(chunks[1], batch).ok());
}

}  // namespace
}  // namespace pih
