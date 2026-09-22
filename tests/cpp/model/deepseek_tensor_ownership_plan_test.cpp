#include "pih/model/deepseek_tensor_ownership_plan.h"

#include <gtest/gtest.h>

namespace pih { namespace {

std::vector<SafetensorsShardBinding> bindings() {
  return {{"embed.weight", "model-00001-of-00048.safetensors"},
          {"layers.0.attn.wq_a.weight", "model-00002-of-00048.safetensors"},
          {"layers.10.ffn.experts.7.w1.weight", "model-00012-of-00048.safetensors"},
          {"layers.11.attn_norm.weight", "model-00013-of-00048.safetensors"},
          {"layers.34.hc_ffn_fn", "model-00036-of-00048.safetensors"},
          {"layers.42.ffn.shared_experts.w2.weight", "model-00044-of-00048.safetensors"},
          {"norm.weight", "model-00045-of-00048.safetensors"},
          {"head.weight", "model-00045-of-00048.safetensors"},
          {"hc_head_scale", "model-00045-of-00048.safetensors"},
          {"mtp.0.main_proj.weight", "model-00046-of-00048.safetensors"},
          {"mtp.1.attn_norm.weight", "model-00047-of-00048.safetensors"},
          {"mtp.2.markov_head.markov_w1.weight", "model-00048-of-00048.safetensors"}};
}

TEST(DeepSeekTensorOwnershipPlanTest, PartitionsEnabledCheckpointAcrossPp4) {
  auto pipeline = DeepSeekPipelinePlan::Create(4, true);
  ASSERT_TRUE(pipeline.ok());
  auto plan = DeepSeekTensorOwnershipPlan::Create(*pipeline, bindings());
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->owned_count(0), 3U);
  EXPECT_EQ(plan->owned_count(1), 1U);
  EXPECT_EQ(plan->owned_count(2), 1U);
  EXPECT_EQ(plan->owned_count(3), 7U);
  EXPECT_EQ(plan->excluded_dspark_count(), 0U);
  for (const auto& record : plan->records()) {
    if (record.tensor_name.starts_with("mtp.")) {
      EXPECT_EQ(record.role, DeepSeekTensorRole::kDspark);
      EXPECT_EQ(record.owner_rank, 3U);
    }
  }
}

TEST(DeepSeekTensorOwnershipPlanTest, ExplicitlyExcludesDsparkInDisabledProfile) {
  auto pipeline = DeepSeekPipelinePlan::Create(4, false);
  ASSERT_TRUE(pipeline.ok());
  auto plan = DeepSeekTensorOwnershipPlan::Create(*pipeline, bindings());
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan->excluded_dspark_count(), 3U);
  EXPECT_EQ(plan->owned_count(3), 5U);
  EXPECT_EQ(plan->records().front().tensor_name, "embed.weight");
  EXPECT_EQ(plan->records().back().tensor_name,
            "norm.weight");
}

TEST(DeepSeekTensorOwnershipPlanTest, CanonicalizesAndRejectsDuplicateInput) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, true).value();
  std::vector<SafetensorsShardBinding> duplicate{
      {"head.weight", "b.safetensors"},
      {"head.weight", "a.safetensors"}};
  EXPECT_FALSE(
      DeepSeekTensorOwnershipPlan::Create(pipeline, duplicate).ok());
}

TEST(DeepSeekTensorOwnershipPlanTest, RejectsAmbiguousOrUnknownNames) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, true).value();
  for (const auto* name : {"layers.01.attn.weight", "layers.43.attn.weight",
                           "layers.1", "mtp.01.weight", "mtp.3.weight",
                           "model.layers.0.weight", "unknown.weight"}) {
    std::vector<SafetensorsShardBinding> one{{name, "a.safetensors"}};
    EXPECT_FALSE(DeepSeekTensorOwnershipPlan::Create(pipeline, one).ok())
        << name;
  }
}

} }  // namespace pih
