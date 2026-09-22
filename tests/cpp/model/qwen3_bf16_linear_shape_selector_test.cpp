#include "pih/model/qwen3_bf16_linear_shape_selector.h"

#include <gtest/gtest.h>

namespace pih {

TEST(QwenBf16LinearShapeSelectorTest, SelectsPrefillAndDecodeShapes) {
  auto selector = QwenBf16LinearShapeSelector::Create(37, 1).value();
  EXPECT_EQ(selector.select(37).value(),
            QwenBf16LinearShapeRole::kPrefill);
  EXPECT_EQ(selector.select(1).value(), QwenBf16LinearShapeRole::kDecode);
  EXPECT_FALSE(selector.select(2).ok());
}

TEST(QwenBf16LinearShapeSelectorTest, OneTokenRequestUsesPrefillRole) {
  auto selector = QwenBf16LinearShapeSelector::Create(1, 1).value();
  EXPECT_EQ(selector.select(1).value(),
            QwenBf16LinearShapeRole::kPrefill);
}

TEST(QwenBf16LinearShapeSelectorTest, RejectsUnboundedOrNonDecodeShape) {
  EXPECT_FALSE(QwenBf16LinearShapeSelector::Create(0, 1).ok());
  EXPECT_FALSE(QwenBf16LinearShapeSelector::Create(4097, 1).ok());
  EXPECT_FALSE(QwenBf16LinearShapeSelector::Create(4, 2).ok());
  EXPECT_FALSE(QwenBf16LinearShapeSelector::Create(4, 5, 1).ok());
}

TEST(QwenBf16LinearShapeSelectorTest, SelectsBoundedTailPrefillShape) {
  auto selector = QwenBf16LinearShapeSelector::Create(4096, 37, 1).value();
  EXPECT_EQ(selector.select(4096).value(),
            QwenBf16LinearShapeRole::kPrefill);
  EXPECT_EQ(selector.select(37).value(),
            QwenBf16LinearShapeRole::kTailPrefill);
  EXPECT_EQ(selector.select(1).value(), QwenBf16LinearShapeRole::kDecode);
}

}  // namespace pih
