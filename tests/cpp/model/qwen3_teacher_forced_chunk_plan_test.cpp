#include <array>
#include <gtest/gtest.h>
#include "pih/model/qwen3_teacher_forced_chunk_plan.h"

namespace pih { namespace {
TEST(QwenTeacherForcedChunkPlanTest, PartitionsEveryCategoryWithoutCrossingBoundaries) {
  const std::array<std::uint64_t,6> rows{100000,100001,100002,100003,100004,100005};
  auto plan=QwenTeacherForcedChunkPlan::Create(rows,32);
  ASSERT_TRUE(plan.ok())<<plan.status().message();
  std::array<std::uint64_t,6> observed{};
  for(const auto& chunk:plan->chunks()){
    const auto category=static_cast<std::size_t>(chunk.category);
    EXPECT_EQ(chunk.category_row_begin,observed[category]);
    EXPECT_GT(chunk.rows,0U);EXPECT_LE(chunk.rows,32U);
    EXPECT_EQ(chunk.logits_bytes,static_cast<std::uint64_t>(chunk.rows)*151936U*4U);
    observed[category]+=chunk.rows;
  }
  EXPECT_EQ(observed,rows);
  EXPECT_EQ(plan->total_rows(),600015U);
}
TEST(QwenTeacherForcedChunkPlanTest, FreezesTailRowsAndCustomWidth) {
  const std::array<std::uint64_t,6> rows{100001,100001,100001,100001,100001,100001};
  auto plan=QwenTeacherForcedChunkPlan::Create(rows,17).value();
  EXPECT_EQ(plan.maximum_rows_per_chunk(),17U);
  EXPECT_EQ(plan.chunks().front().rows,17U);
  EXPECT_EQ(plan.chunks()[5882].rows,7U);
}
TEST(QwenTeacherForcedChunkPlanTest, RejectsIncompleteOrUnsafeCoverage) {
  std::array<std::uint64_t,6> rows{};rows.fill(100000);
  rows[2]=99999;EXPECT_FALSE(QwenTeacherForcedChunkPlan::Create(rows).ok());
  rows[2]=100000;EXPECT_FALSE(QwenTeacherForcedChunkPlan::Create(rows,0).ok());
  EXPECT_FALSE(QwenTeacherForcedChunkPlan::Create(rows,33).ok());
}
} }
