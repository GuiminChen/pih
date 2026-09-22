#include "pih/model/deepseek_mhc_submission_assembler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekMhcWeightBindings weights() {
  return {0x101, 0x102, 0x103, 0x104,
          0x105, 0x106, 0x107, 0x108, 79};
}

DeepSeekMhcDeviceView workspace() {
  return {0x201, 0x202, 0x203, 0x204, 0x205, 0x206};
}

TEST(DeepSeekMhcSubmissionAssemblerTest,
     AssemblesOfficialAttentionThenFeedForwardPingPong) {
  auto result = DeepSeekMhcSubmissionAssembler::Assemble(
      7, weights(), workspace(), 8, 4, 0x201, 0x301, 0x401, 0x501);
  ASSERT_TRUE(result.ok()) << result.status().message();

  EXPECT_EQ(result->layer_input_bf16, 0x203U);
  EXPECT_EQ(result->residual_output_bf16, 0x201U);
  EXPECT_EQ(result->attention.kind, DeepSeekMhcBranchKind::kAttention);
  EXPECT_EQ(result->attention.residual_bf16, 0x201U);
  EXPECT_EQ(result->attention.norm_weight_bf16, 0x101U);
  EXPECT_EQ(result->attention.fn_f32, 0x102U);
  EXPECT_EQ(result->attention.branch_output_bf16, 0x301U);
  EXPECT_EQ(result->attention.output_bf16, 0x202U);
  EXPECT_EQ(result->attention.post_mix_f32, 0x205U);
  EXPECT_EQ(result->attention.device_error_flag_u32, 0x401U);

  EXPECT_EQ(result->feed_forward.kind,
            DeepSeekMhcBranchKind::kFeedForward);
  EXPECT_EQ(result->feed_forward.residual_bf16, 0x202U);
  EXPECT_EQ(result->feed_forward.norm_weight_bf16, 0x105U);
  EXPECT_EQ(result->feed_forward.fn_f32, 0x106U);
  EXPECT_EQ(result->feed_forward.branch_output_bf16, 0x204U);
  EXPECT_EQ(result->feed_forward.output_bf16, 0x201U);
  EXPECT_EQ(result->feed_forward.stream, 0x501U);
  EXPECT_EQ(result->feed_forward.sinkhorn_iterations, 20U);
}

TEST(DeepSeekMhcSubmissionAssemblerTest,
     AcceptsExternalFirstResidualAndRejectsUnsafeAliasOrCapacity) {
  EXPECT_TRUE(DeepSeekMhcSubmissionAssembler::Assemble(
                  0, weights(), workspace(), 8, 4, 0x601, 0x301,
                  0x401, 0x501)
                  .ok());
  EXPECT_FALSE(DeepSeekMhcSubmissionAssembler::Assemble(
                   44, weights(), workspace(), 8, 4, 0x601, 0x301,
                   0x401, 0x501)
                   .ok());
  EXPECT_FALSE(DeepSeekMhcSubmissionAssembler::Assemble(
                   7, weights(), workspace(), 8, 9, 0x201, 0x301,
                   0x401, 0x501)
                   .ok());
  EXPECT_FALSE(DeepSeekMhcSubmissionAssembler::Assemble(
                   7, weights(), workspace(), 8, 4, 0x202, 0x301,
                   0x401, 0x501)
                   .ok());
  auto incomplete = weights();
  incomplete.feed_forward_norm_bf16 = 0;
  EXPECT_FALSE(DeepSeekMhcSubmissionAssembler::Assemble(
                   7, incomplete, workspace(), 8, 4, 0x201, 0x301,
                   0x401, 0x501)
                   .ok());
}

}  // namespace
}  // namespace pih
