#include "pih/model/deepseek_fused_kernel_plan.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekOptimizationPolicy policy(DeepSeekFusedKernelSet set,
                                  DeepSeekGpuArchitecture architecture) {
  return DeepSeekOptimizationPolicy::Create({
      .architecture = architecture,
      .world_size = 2,
      .chunked_prefill = true,
      .attention_layout = DeepSeekAttentionPhysicalLayoutVersion::kV1,
      .fused_kernel_set = set,
  }).value();
}

DeepSeekFusedKernelManifest manifest(DeepSeekGpuArchitecture architecture) {
  return {
      .logical_id = "deepseek.flash0731.fused.v1",
      .architecture = architecture,
      .maximum_tokens = 4096,
      .hidden_size = 4096,
      .routed_experts = 256,
      .activated_experts = 6,
      .cubin_digest = Sha256Digest::ParseHex(std::string(64, 'a')).value(),
      .parameter_abi_root =
          DeepSeekFusedKernelPlan::ExpectedAbiRoot(architecture).value(),
  };
}

TEST(DeepSeekFusedKernelPlanTest, PreservesExplicitUnfusedControlPath) {
  auto plan = DeepSeekFusedKernelPlan::Compile(
      policy(DeepSeekFusedKernelSet::kUnfusedControl,
             DeepSeekGpuArchitecture::kSm89),
      512, nullptr, nullptr);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_FALSE(plan->fused());
  EXPECT_EQ(plan->logical_id(), "deepseek.flash0731.unfused.control.v1");
}

TEST(DeepSeekFusedKernelPlanTest, SelectsOnlyExactManifestAndArtifact) {
  auto selected = manifest(DeepSeekGpuArchitecture::kSm90);
  auto observed = selected.cubin_digest;
  auto plan = DeepSeekFusedKernelPlan::Compile(
      policy(DeepSeekFusedKernelSet::kVerifiedV1,
             DeepSeekGpuArchitecture::kSm90),
      4096, &selected, &observed);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_TRUE(plan->fused());
  EXPECT_EQ(plan->logical_id(), selected.logical_id);
  EXPECT_EQ(plan->cubin_digest(), observed);
  EXPECT_EQ(plan->identity().hex().size(), 64U);
}

TEST(DeepSeekFusedKernelPlanTest,
     RejectsArchitectureShapeCubinAndAbiDriftBeforeSelection) {
  auto selected = manifest(DeepSeekGpuArchitecture::kSm89);
  auto observed = selected.cubin_digest;
  const auto fused = policy(DeepSeekFusedKernelSet::kVerifiedV1,
                            DeepSeekGpuArchitecture::kSm89);
  auto wrong_architecture = selected;
  wrong_architecture.architecture = DeepSeekGpuArchitecture::kSm90;
  EXPECT_FALSE(DeepSeekFusedKernelPlan::Compile(
      fused, 512, &wrong_architecture, &observed).ok());
  EXPECT_FALSE(DeepSeekFusedKernelPlan::Compile(
      fused, 4097, &selected, &observed).ok());
  auto wrong_cubin = Sha256Digest::ParseHex(std::string(64, 'b')).value();
  EXPECT_FALSE(DeepSeekFusedKernelPlan::Compile(
      fused, 512, &selected, &wrong_cubin).ok());
  selected.parameter_abi_root =
      Sha256Digest::ParseHex(std::string(64, 'c')).value();
  EXPECT_FALSE(DeepSeekFusedKernelPlan::Compile(
      fused, 512, &selected, &observed).ok());
}

}  // namespace
}  // namespace pih
