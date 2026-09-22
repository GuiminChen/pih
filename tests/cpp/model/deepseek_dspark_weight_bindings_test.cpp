#include "pih/model/deepseek_dspark_weight_bindings.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "deepseek_dspark_weight_test_fixture.h"

namespace pih { namespace {

TEST(DeepSeekDsparkWeightBindingsTest,
     BindsThreeCommonStagesAndExactBoundaryOwners) {
  std::vector<std::string> requested;
  auto bindings = DeepSeekDsparkWeightBindings::Resolve(
      test::deepseek_dspark_test_stage(),
      [&requested](std::string_view name) {
        return test::deepseek_dspark_test_tensor(name, &requested);
      });
  ASSERT_TRUE(bindings.ok()) << bindings.status().message();
  EXPECT_EQ(bindings->generation(), 29U);
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    auto stage = deepseek_dspark_stage_id(index).value();
    EXPECT_EQ(bindings->common(stage).stage, stage);
    EXPECT_NE(bindings->common(stage).attention.wq_a_fp8, 0U);
    EXPECT_NE(bindings->common(stage).mhc.feed_forward_fn_f32, 0U);
    EXPECT_NE(bindings->common(stage).router_weight_bf16, 0U);
    EXPECT_NE(bindings->common(stage).router_bias_f32, 0U);
    EXPECT_NE(bindings->common(stage).shared_expert.w2.packed.address, 0U);
  }
  EXPECT_NE(bindings->boundary().main_proj_fp8, 0U);
  EXPECT_NE(bindings->head().markov_embedding_bf16, 0U);
  EXPECT_NE(bindings->head().confidence_weight_bf16, 0U);
  EXPECT_EQ(std::ranges::count(requested, "mtp.0.main_norm.weight"), 1);
  EXPECT_EQ(std::ranges::count(requested, "mtp.1.main_norm.weight"), 0);
  EXPECT_EQ(std::ranges::count(requested, "mtp.2.norm.weight"), 1);
  EXPECT_EQ(std::ranges::count(requested, "mtp.0.norm.weight"), 0);
}

TEST(DeepSeekDsparkWeightBindingsTest,
     RejectsNonOwningStageAndWrongStage2HeadShape) {
  auto first = test::deepseek_dspark_test_stage();
  first.owns_dspark = false;
  EXPECT_FALSE(DeepSeekDsparkWeightBindings::Resolve(
      first, [](std::string_view name) {
        return test::deepseek_dspark_test_tensor(name);
      }).ok());
  EXPECT_FALSE(DeepSeekDsparkWeightBindings::Resolve(
      test::deepseek_dspark_test_stage(), [](std::string_view name) {
        return test::deepseek_dspark_test_tensor(name, nullptr, true);
      }).ok());
}

TEST(DeepSeekDsparkWeightBindingsTest,
     RejectsMixedGenerationOutsideAttentionAndMhc) {
  EXPECT_FALSE(DeepSeekDsparkWeightBindings::Resolve(
      test::deepseek_dspark_test_stage(), [](std::string_view name) {
        const auto generation =
            name == "mtp.1.ffn.gate.weight" ? 31U : 29U;
        return test::deepseek_dspark_test_tensor(
            name, nullptr, false, generation);
      }).ok());
  EXPECT_FALSE(DeepSeekDsparkWeightBindings::Resolve(
      test::deepseek_dspark_test_stage(), [](std::string_view name) {
        const auto generation =
            name == "mtp.2.markov_head.markov_w1.weight" ? 31U : 29U;
        return test::deepseek_dspark_test_tensor(
            name, nullptr, false, generation);
      }).ok());
}

} }  // namespace pih
