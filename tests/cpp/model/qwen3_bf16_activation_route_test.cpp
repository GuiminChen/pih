#include "pih/model/qwen3_bf16_activation_route.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Qwen3Config official_config() {
  return Qwen3Config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                     1'000'000.0, 0.000001, 151643, 151645};
}

TEST(QwenBf16ActivationRoutePlanTest, FreezesAttentionDataflowAndAliases) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  ASSERT_TRUE(schedule.ok());
  auto plan = QwenBf16ActivationRoutePlan::Create(*schedule);
  ASSERT_TRUE(plan.ok());

  const auto& prepare = (*plan)[1];
  EXPECT_EQ(prepare.reads[0], QwenBf16ActivationSlot::kPositions);
  EXPECT_EQ(prepare.writes[0], QwenBf16ActivationSlot::kRopeCosine);
  EXPECT_EQ(prepare.writes[1], QwenBf16ActivationSlot::kRopeSine);

  const auto& rope = (*plan)[8];
  EXPECT_EQ(rope.read_count, 4);
  EXPECT_EQ(rope.reads[0], QwenBf16ActivationSlot::kQuery);
  EXPECT_EQ(rope.reads[1], QwenBf16ActivationSlot::kKey);
  EXPECT_EQ(rope.reads[2], QwenBf16ActivationSlot::kRopeCosine);
  EXPECT_EQ(rope.reads[3], QwenBf16ActivationSlot::kRopeSine);
  EXPECT_EQ(rope.write_count, 2);
  EXPECT_EQ(rope.writes[0], QwenBf16ActivationSlot::kQuery);
  EXPECT_EQ(rope.writes[1], QwenBf16ActivationSlot::kKey);

  const auto& append = (*plan)[9];
  EXPECT_EQ(append.reads[3], QwenBf16ActivationSlot::kKvAppendHandles);
  EXPECT_EQ(append.reads[4], QwenBf16ActivationSlot::kKvTokenOffsets);

  const auto& paged_gqa = (*plan)[10];
  EXPECT_EQ(paged_gqa.reads[3],
            QwenBf16ActivationSlot::kKvVisibleHandles);

  const auto& attention_residual = (*plan)[12];
  EXPECT_EQ(attention_residual.reads[0], QwenBf16ActivationSlot::kHidden);
  EXPECT_EQ(attention_residual.reads[1], QwenBf16ActivationSlot::kNormalized);
  EXPECT_EQ(attention_residual.writes[0], QwenBf16ActivationSlot::kHidden);
}

TEST(QwenBf16ActivationRoutePlanTest, UsesOneRowLogitsAndExternalSampleSlot) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  ASSERT_TRUE(schedule.ok());
  auto plan = QwenBf16ActivationRoutePlan::Create(*schedule);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ((*plan)[479].reads[0], QwenBf16ActivationSlot::kNormalized);
  EXPECT_EQ((*plan)[479].writes[0], QwenBf16ActivationSlot::kLogits);
  EXPECT_EQ((*plan)[480].reads[0], QwenBf16ActivationSlot::kLogits);
  EXPECT_EQ((*plan)[480].writes[0], QwenBf16ActivationSlot::kSampledToken);
  EXPECT_EQ((*plan)[480].writes[1], QwenBf16ActivationSlot::kDeviceError);
}

TEST(QwenBf16ActivationRoutePlanTest, EveryStepHasBoundedReadsAndWrites) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  ASSERT_TRUE(schedule.ok());
  auto plan = QwenBf16ActivationRoutePlan::Create(*schedule);
  ASSERT_TRUE(plan.ok());
  for (std::size_t step = 0; step < schedule->size(); ++step) {
    EXPECT_GT((*plan)[step].read_count, 0) << step;
    EXPECT_GT((*plan)[step].write_count, 0) << step;
    EXPECT_LE((*plan)[step].read_count,
              QwenBf16ActivationRoute::kMaximumReads);
    EXPECT_LE((*plan)[step].write_count,
              QwenBf16ActivationRoute::kMaximumWrites);
  }
}

}  // namespace
}  // namespace pih
