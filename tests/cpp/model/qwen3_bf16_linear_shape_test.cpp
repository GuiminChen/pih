#include "pih/model/qwen3_bf16_linear_shape.h"

#include <array>
#include <cstdint>
#include <string_view>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenBf16LinearShapeTest, FreezesEveryOfficialWeightOrientation) {
  struct Expected final {
    QwenBf16LinearKind kind;
    std::uint64_t input;
    std::uint64_t output;
    std::string_view id;
  };
  constexpr std::array expected{
      Expected{QwenBf16LinearKind::kQuery, 1024, 2048,
               "qwen.linear.query.bf16.v1"},
      Expected{QwenBf16LinearKind::kKey, 1024, 1024,
               "qwen.linear.key.bf16.v1"},
      Expected{QwenBf16LinearKind::kValue, 1024, 1024,
               "qwen.linear.value.bf16.v1"},
      Expected{QwenBf16LinearKind::kAttentionOutput, 2048, 1024,
               "qwen.linear.attention_output.bf16.v1"},
      Expected{QwenBf16LinearKind::kGate, 1024, 3072,
               "qwen.linear.gate.bf16.v1"},
      Expected{QwenBf16LinearKind::kUp, 1024, 3072,
               "qwen.linear.up.bf16.v1"},
      Expected{QwenBf16LinearKind::kDown, 3072, 1024,
               "qwen.linear.down.bf16.v1"},
      Expected{QwenBf16LinearKind::kLmHead, 1024, 151936,
               "qwen.linear.lm_head.bf16.v1"}};
  for (const auto& item : expected) {
    auto shape = QwenBf16LinearShape::Create(item.kind, 16);
    ASSERT_TRUE(shape.ok()) << static_cast<int>(item.kind);
    EXPECT_EQ(shape->tokens(), 16);
    EXPECT_EQ(shape->input_features(), item.input);
    EXPECT_EQ(shape->output_features(), item.output);
    EXPECT_EQ(shape->weight_rows(), item.output);
    EXPECT_EQ(shape->weight_columns(), item.input);
    EXPECT_EQ(shape->output_dtype(),
              item.kind == QwenBf16LinearKind::kLmHead ? DType::kFloat32
                                                       : DType::kBFloat16);
    EXPECT_EQ(shape->logical_id(), item.id);
  }
}

TEST(QwenBf16LinearShapeTest, RejectsUnboundedTokensAndUnknownKinds) {
  EXPECT_FALSE(QwenBf16LinearShape::Create(
                   QwenBf16LinearKind::kQuery, 0)
                   .ok());
  EXPECT_FALSE(QwenBf16LinearShape::Create(
                   QwenBf16LinearKind::kQuery,
                   QwenBf16LinearShape::kMaximumTokensPerPlan + 1)
                   .ok());
  EXPECT_FALSE(QwenBf16LinearShape::Create(
                   static_cast<QwenBf16LinearKind>(255), 1)
                   .ok());
}

TEST(QwenBf16LinearShapeTest, LmHeadConsumesOnlyActivePackedSampleRows) {
  auto hidden_rows = qwen_bf16_linear_execution_rows(
      QwenBf16LinearKind::kQuery, 4096, 1);
  auto logit_rows = qwen_bf16_linear_execution_rows(
      QwenBf16LinearKind::kLmHead, 4096, 1);
  ASSERT_TRUE(hidden_rows.ok());
  ASSERT_TRUE(logit_rows.ok());
  EXPECT_EQ(*hidden_rows, 4096);
  EXPECT_EQ(*logit_rows, 1);
  auto packed_logits = qwen_bf16_linear_execution_rows(
      QwenBf16LinearKind::kLmHead, 4096, 2);
  ASSERT_TRUE(packed_logits.ok());
  EXPECT_EQ(*packed_logits, 2);
  EXPECT_FALSE(qwen_bf16_linear_execution_rows(
                   QwenBf16LinearKind::kLmHead, 2, 3).ok());
}

}  // namespace
}  // namespace pih
