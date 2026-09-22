#include <cstdint>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_capacity_accounting.h"

namespace pih {
namespace {

TEST(QwenSemanticCapacityAccountingTest, SeparatesOverlappingOwnerPeaks) {
  auto pair = qwen_semantic_capacity_pair(1000, 100, 200, 300, 500);
  ASSERT_TRUE(pair.ok());
  EXPECT_EQ(pair->instrumented_device_peak_bytes, 1200);
  EXPECT_EQ(pair->instrumented_pinned_peak_bytes, 600);
  EXPECT_EQ(pair->control_device_peak_bytes, 1000);
  EXPECT_EQ(pair->control_pinned_peak_bytes, 600);

  pair = qwen_semantic_capacity_pair(1000, 100, 200, 700, 500);
  ASSERT_TRUE(pair.ok());
  EXPECT_EQ(pair->instrumented_pinned_peak_bytes, 800);
  EXPECT_EQ(pair->control_pinned_peak_bytes, 600);
}

TEST(QwenSemanticCapacityAccountingTest, RejectsMissingOwnerAndOverflow) {
  EXPECT_FALSE(qwen_semantic_capacity_pair(1000, 100, 0, 300, 500).ok());
  auto overflow = qwen_semantic_capacity_pair(
      UINT64_MAX, 100, 1, 1, 1);
  ASSERT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.status().code(), StatusCode::kResourceExhausted);
}

}  // namespace
}  // namespace pih
