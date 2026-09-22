#include "pih/model/qwen3_bf16_engine_resource_plan.h"

#include <gtest/gtest.h>

#include "pih/model/qwen3_manifest.h"

namespace pih {
namespace {

TEST(QwenBf16EngineResourcePlanTest, SeparatesChunkAndContextResidency) {
  auto plan = QwenBf16EngineResourcePlan::Create(4096, 40960, 2560, 1 << 20);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->maximum_step_tokens(), 4096);
  EXPECT_EQ(plan->maximum_sequence_tokens(), 40960);
  EXPECT_EQ(plan->step_staging_bytes(), 126976);
  EXPECT_EQ(plan->activation_bytes(), UINT64_C(67108864));
  EXPECT_EQ(plan->mlp_bytes(), UINT64_C(50331648));
  EXPECT_EQ(plan->rope_bytes(), UINT64_C(2097152));
  EXPECT_EQ(plan->sampler_workspace_bytes(), UINT64_C(607744));
  EXPECT_EQ(plan->logits_bytes(), UINT64_C(1215488));
  EXPECT_EQ(plan->kv_backing_bytes(),
            UINT64_C(2560) * QwenKvSlotPool::kSlotPayloadBytes);
  EXPECT_EQ(plan->kv_metadata_bytes(), UINT64_C(2560) * 16);
  EXPECT_EQ(plan->resident_weight_bytes(),
            Qwen3Manifest::kOfficialSourcePayloadBytes);
  EXPECT_GT(plan->total_device_bytes(), plan->resident_weight_bytes());
}

TEST(QwenBf16EngineResourcePlanTest, RejectsUnfundedOrInvalidBounds) {
  EXPECT_FALSE(QwenBf16EngineResourcePlan::Create(0, 40960, 2560, 0).ok());
  EXPECT_FALSE(QwenBf16EngineResourcePlan::Create(4097, 40960, 2560, 0).ok());
  EXPECT_FALSE(QwenBf16EngineResourcePlan::Create(4096, 4095, 2560, 0).ok());
  EXPECT_FALSE(QwenBf16EngineResourcePlan::Create(4096, 40960, 2559, 0).ok());
  EXPECT_FALSE(QwenBf16EngineResourcePlan::Create(1, 40961, 2561, 0).ok());
  EXPECT_FALSE(
      QwenBf16EngineResourcePlan::Create(4096, 40960, 2560, 0, 0).ok());
  EXPECT_FALSE(
      QwenBf16EngineResourcePlan::Create(32, 40960, 2560, 0, 33).ok());
}

TEST(QwenBf16EngineResourcePlanTest, ReservesBoundedPackedBatchArenas) {
  constexpr std::uint32_t kMaximumBatchSequences = 32;
  auto plan = QwenBf16EngineResourcePlan::Create(
      4096, 40960, 2560, 1 << 20, kMaximumBatchSequences);
  ASSERT_TRUE(plan.ok()) << plan.status().message();

  EXPECT_EQ(plan->maximum_batch_sequences(), kMaximumBatchSequences);
  EXPECT_EQ(plan->execution_layout().active_logit_rows(),
            kMaximumBatchSequences);
  EXPECT_EQ(plan->sampled_result_bytes(),
            plan->packed_result_layout().device_error().offset_bytes);
  EXPECT_EQ(plan->sampled_token_bytes(), plan->sampled_result_bytes());
  EXPECT_GE(plan->step_staging_bytes(),
            plan->packed_staging_layout().total_bytes());
  EXPECT_GE(plan->pinned_result_bytes(),
            plan->packed_result_layout().total_bytes());
  EXPECT_EQ(plan->packed_result_layout().sample_capacity(),
            kMaximumBatchSequences);
  EXPECT_EQ(plan->logits_bytes(),
            static_cast<std::uint64_t>(2) * kMaximumBatchSequences *
                QwenBf16ExecutionArenaLayout::kVocabularySize *
                sizeof(float));
  EXPECT_EQ(plan->sampler_workspace_bytes(),
            static_cast<std::uint64_t>(kMaximumBatchSequences) *
                QwenBf16ExecutionArenaLayout::kVocabularySize *
                sizeof(std::uint32_t));
}

}  // namespace
}  // namespace pih
