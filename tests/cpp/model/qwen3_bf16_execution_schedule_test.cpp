#include "pih/model/qwen3_bf16_execution_schedule.h"

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

namespace pih {
namespace {

Qwen3Config official_config() {
  return Qwen3Config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                     1'000'000.0, 0.000001, 151643, 151645};
}

constexpr std::array kLayerOps{
    QwenBf16ExecutionOp::kInputRmsNorm,
    QwenBf16ExecutionOp::kQueryLinear,
    QwenBf16ExecutionOp::kKeyLinear,
    QwenBf16ExecutionOp::kValueLinear,
    QwenBf16ExecutionOp::kQueryRmsNorm,
    QwenBf16ExecutionOp::kKeyRmsNorm,
    QwenBf16ExecutionOp::kRope,
    QwenBf16ExecutionOp::kKvAppend,
    QwenBf16ExecutionOp::kPagedGqa,
    QwenBf16ExecutionOp::kAttentionOutputLinear,
    QwenBf16ExecutionOp::kAttentionResidual,
    QwenBf16ExecutionOp::kPostAttentionRmsNorm,
    QwenBf16ExecutionOp::kGateLinear,
    QwenBf16ExecutionOp::kUpLinear,
    QwenBf16ExecutionOp::kSiluMul,
    QwenBf16ExecutionOp::kDownLinear,
    QwenBf16ExecutionOp::kMlpResidual,
};

TEST(QwenBf16ExecutionScheduleTest, FreezesAllTwentyEightLayersInOrder) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  ASSERT_TRUE(schedule.ok()) << schedule.status().message();
  EXPECT_EQ(schedule->size(), 481);
  EXPECT_EQ((*schedule)[0], (QwenBf16ExecutionStep{
                                QwenBf16ExecutionOp::kEmbedding,
                                QwenBf16ExecutionSchedule::kGlobalLayer}));

  for (std::uint32_t layer = 0; layer < 28; ++layer) {
    const std::size_t base = 2 + layer * kLayerOps.size();
    for (std::size_t operation = 0; operation < kLayerOps.size(); ++operation) {
      EXPECT_EQ((*schedule)[base + operation],
                (QwenBf16ExecutionStep{kLayerOps[operation], layer}));
    }
  }

  EXPECT_EQ((*schedule)[478].operation, QwenBf16ExecutionOp::kFinalRmsNorm);
  EXPECT_EQ((*schedule)[479].operation, QwenBf16ExecutionOp::kLmHead);
  EXPECT_EQ((*schedule)[480].operation, QwenBf16ExecutionOp::kGreedyArgmax);
  EXPECT_EQ((*schedule)[480].layer, QwenBf16ExecutionSchedule::kGlobalLayer);
}

TEST(QwenBf16ExecutionScheduleTest, MapsOnlyLinearOperationsToWeightKinds) {
  struct Expected final {
    QwenBf16ExecutionOp operation;
    QwenBf16LinearKind kind;
  };
  constexpr std::array expected{
      Expected{QwenBf16ExecutionOp::kQueryLinear, QwenBf16LinearKind::kQuery},
      Expected{QwenBf16ExecutionOp::kKeyLinear, QwenBf16LinearKind::kKey},
      Expected{QwenBf16ExecutionOp::kValueLinear, QwenBf16LinearKind::kValue},
      Expected{QwenBf16ExecutionOp::kAttentionOutputLinear,
               QwenBf16LinearKind::kAttentionOutput},
      Expected{QwenBf16ExecutionOp::kGateLinear, QwenBf16LinearKind::kGate},
      Expected{QwenBf16ExecutionOp::kUpLinear, QwenBf16LinearKind::kUp},
      Expected{QwenBf16ExecutionOp::kDownLinear, QwenBf16LinearKind::kDown},
      Expected{QwenBf16ExecutionOp::kLmHead, QwenBf16LinearKind::kLmHead},
  };
  for (const auto& item : expected) {
    auto kind = qwen_bf16_linear_kind(item.operation);
    ASSERT_TRUE(kind.ok());
    EXPECT_EQ(*kind, item.kind);
  }
  EXPECT_FALSE(qwen_bf16_linear_kind(QwenBf16ExecutionOp::kRope).ok());
}

TEST(QwenBf16ExecutionScheduleTest, RejectsAnyOfficialArchitectureDrift) {
  auto config = official_config();
  config.hidden_size = 2048;
  EXPECT_FALSE(QwenBf16ExecutionSchedule::Create(config).ok());
  config = official_config();
  config.layers = 27;
  EXPECT_FALSE(QwenBf16ExecutionSchedule::Create(config).ok());
  config = official_config();
  config.rms_norm_epsilon = 0.00001;
  EXPECT_FALSE(QwenBf16ExecutionSchedule::Create(config).ok());
  config = official_config();
  config.eos_token_id = 151644;
  EXPECT_FALSE(QwenBf16ExecutionSchedule::Create(config).ok());
}

}  // namespace
}  // namespace pih
