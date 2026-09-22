#include "pih/model/deepseek_optimization_policy.h"

#include <gtest/gtest.h>

namespace pih { namespace {

DeepSeekOptimizationProfile baseline() {
  return {
      .architecture = DeepSeekGpuArchitecture::kSm90,
      .world_size = 4,
      .chunked_prefill = true,
      .attention_layout = DeepSeekAttentionPhysicalLayoutVersion::kV1,
      .fused_kernel_set = DeepSeekFusedKernelSet::kVerifiedV1,
      .observed_miss_debit = true,
      .dspark = false,
      .cuda_graph = false,
      .expert_locality_reorder = false,
  };
}

TEST(DeepSeekOptimizationPolicyTest, RejectsForbiddenProfileFeatures) {
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    auto dspark = baseline();
    dspark.architecture = DeepSeekGpuArchitecture::kSm89;
    dspark.world_size = world_size;
    dspark.dspark = true;
    auto rejected = DeepSeekOptimizationPolicy::Create(dspark);
    ASSERT_FALSE(rejected.ok());
    EXPECT_EQ(rejected.status().code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(rejected.status().message(),
              "DeepSeek DSpark production is restricted to SM90");

    dspark.architecture = DeepSeekGpuArchitecture::kSm90;
    auto accepted = DeepSeekOptimizationPolicy::Create(dspark);
    ASSERT_TRUE(accepted.ok()) << accepted.status().message();
    EXPECT_TRUE(accepted->dspark());
  }
  auto graph = baseline();
  graph.cuda_graph = true;
  EXPECT_EQ(DeepSeekOptimizationPolicy::Create(graph).status().code(),
            StatusCode::kFailedPrecondition);
  auto reordered = baseline();
  reordered.expert_locality_reorder = true;
  EXPECT_EQ(DeepSeekOptimizationPolicy::Create(reordered).status().code(),
            StatusCode::kFailedPrecondition);
}

TEST(DeepSeekOptimizationPolicyTest, FreezesClosedProfileAndStableIdentity) {
  auto first = DeepSeekOptimizationPolicy::Create(baseline());
  auto second = DeepSeekOptimizationPolicy::Create(baseline());
  ASSERT_TRUE(first.ok()) << first.status().message();
  ASSERT_TRUE(second.ok()) << second.status().message();
  EXPECT_EQ(first->identity(), second->identity());
  EXPECT_EQ(first->world_size(), 4U);
  EXPECT_TRUE(first->chunked_prefill());
  EXPECT_TRUE(first->observed_miss_debit());
  EXPECT_FALSE(first->dspark());
  EXPECT_FALSE(first->cuda_graph());
  EXPECT_FALSE(first->expert_locality_reorder());
  auto changed = baseline();
  changed.fused_kernel_set = DeepSeekFusedKernelSet::kUnfusedControl;
  auto control = DeepSeekOptimizationPolicy::Create(changed);
  ASSERT_TRUE(control.ok());
  EXPECT_NE(first->identity(), control->identity());
}

TEST(DeepSeekOptimizationPolicyTest,
     DistinguishesAdmittedHardwareAndPipelineAxes) {
  DeepSeekOptimizationProfile rtx_4090_pp1{
      .architecture = DeepSeekGpuArchitecture::kSm89,
      .world_size = 1,
      .fused_kernel_set = DeepSeekFusedKernelSet::kUnfusedControl,
  };
  DeepSeekOptimizationProfile h100_pp4 = rtx_4090_pp1;
  h100_pp4.architecture = DeepSeekGpuArchitecture::kSm90;
  h100_pp4.world_size = 4;
  h100_pp4.dspark = true;

  auto first = DeepSeekOptimizationPolicy::Create(rtx_4090_pp1);
  auto second = DeepSeekOptimizationPolicy::Create(h100_pp4);
  ASSERT_TRUE(first.ok()) << first.status().message();
  ASSERT_TRUE(second.ok()) << second.status().message();
  EXPECT_EQ(first->architecture(), DeepSeekGpuArchitecture::kSm89);
  EXPECT_EQ(first->world_size(), 1U);
  EXPECT_FALSE(first->dspark());
  EXPECT_EQ(second->architecture(), DeepSeekGpuArchitecture::kSm90);
  EXPECT_EQ(second->world_size(), 4U);
  EXPECT_TRUE(second->dspark());
  EXPECT_NE(first->identity(), second->identity());
}

TEST(DeepSeekOptimizationPolicyTest, RejectsInvalidTopology) {
  for (const auto world_size : {0U, 5U}) {
    auto profile = baseline();
    profile.world_size = world_size;
    EXPECT_EQ(DeepSeekOptimizationPolicy::Create(profile).status().code(),
              StatusCode::kInvalidArgument);
  }
}

TEST(DeepSeekOptimizationPolicyTest, RejectsUnknownClosedEnumValues) {
  auto architecture = baseline();
  architecture.architecture = static_cast<DeepSeekGpuArchitecture>(7);
  EXPECT_EQ(DeepSeekOptimizationPolicy::Create(architecture).status().code(),
            StatusCode::kInvalidArgument);
  auto layout = baseline();
  layout.attention_layout =
      static_cast<DeepSeekAttentionPhysicalLayoutVersion>(7);
  EXPECT_EQ(DeepSeekOptimizationPolicy::Create(layout).status().code(),
            StatusCode::kInvalidArgument);
  auto fused = baseline();
  fused.fused_kernel_set = static_cast<DeepSeekFusedKernelSet>(7);
  EXPECT_EQ(DeepSeekOptimizationPolicy::Create(fused).status().code(),
            StatusCode::kInvalidArgument);
}

} }  // namespace pih
