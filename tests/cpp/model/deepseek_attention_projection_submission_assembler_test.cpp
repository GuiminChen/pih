#include "pih/model/deepseek_attention_projection_submission_assembler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekAttentionWeightBindings weights() {
  return {0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107,
          0x108, 0x109, 0x10A, 0x10B, 0x10C, 0x10D, 71};
}

DeepSeekAttentionProjectionDeviceView workspace() {
  return {0x201, 0x202, 0x203, 0x204, 0x205, 0x206, 0x207, 0x208,
          0x209, 0x20A, 0x20B, 0x20C, 0x20D, 0x20E, 0x20F, 0x305,
          0x304, 0x210};
}

TEST(DeepSeekAttentionProjectionSubmissionAssemblerTest,
     AssemblesCheckpointNativeProjectionDataflow) {
  auto assembled = DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
      weights(), workspace(), 8, 4, 0x301, 0x302, 0x303, 104, 0x8);
  ASSERT_TRUE(assembled.ok()) << assembled.status().message();

  EXPECT_EQ(assembled->sparse_query_bf16, 0x207U);
  EXPECT_EQ(assembled->sparse_kv_bf16, 0x208U);
  EXPECT_EQ(assembled->sparse_output_bf16, 0x302U);
  EXPECT_EQ(assembled->branch_output_bf16, 0x20FU);

  const auto& input = assembled->input;
  EXPECT_EQ(input.input_quant.input_bf16, 0x301U);
  EXPECT_EQ(input.input_quant.output_e4m3, 0x201U);
  EXPECT_EQ(input.wq_a.weight_e4m3, 0x101U);
  EXPECT_EQ(input.q_norm.weight_bf16, 0x103U);
  EXPECT_EQ(input.wq_b.output_bf16, 0x207U);
  EXPECT_EQ(input.q_rope.frequencies_f32, 0x303U);
  EXPECT_EQ(input.q_rope.positions_u32, 0x304U);
  EXPECT_EQ(input.q_rope.table_position_count, 104U);
  EXPECT_EQ(input.wkv.output_bf16, 0x208U);
  EXPECT_EQ(input.kv_simulate.kv_bf16, 0x208U);
  EXPECT_EQ(input.input_quant.token_count, 4U);
  EXPECT_EQ(input.input_quant.stream, 0x8U);

  const auto& output = assembled->output;
  EXPECT_TRUE(output.inverse_rope.inverse);
  EXPECT_EQ(output.inverse_rope.input_bf16, 0x302U);
  EXPECT_EQ(output.wo_a.activation_e4m3, 0x20AU);
  EXPECT_EQ(output.wo_a.activation_scale_bits, 0x20BU);
  EXPECT_EQ(output.wo_a.weight_e4m3, 0x10AU);
  EXPECT_EQ(output.wo_a.output_bf16, 0x20CU);
  EXPECT_EQ(output.quant.output_e4m3, 0x20DU);
  EXPECT_EQ(output.wo_b.weight_e4m3, 0x10CU);
  EXPECT_EQ(output.wo_b.output_bf16, 0x20FU);
  EXPECT_EQ(output.wo_b.error_flag, 0x210U);
}

TEST(DeepSeekAttentionProjectionSubmissionAssemblerTest,
     RejectsCapacityOverflowAndIncompleteBorrowedInputs) {
  auto invalid_weights = weights();
  invalid_weights.generation = 0;
  EXPECT_FALSE(DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
                   invalid_weights, workspace(), 8, 4, 0x301, 0x302, 0x303,
                   104, 0x8)
                   .ok());
  EXPECT_FALSE(DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
                   weights(), workspace(), 8, 9, 0x301, 0x302, 0x303,
                   104, 0x8)
                   .ok());
  EXPECT_FALSE(DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
                   weights(), workspace(), 8, 4, 0, 0x302, 0x303,
                   104, 0x8)
                   .ok());
  auto invalid_workspace = workspace();
  invalid_workspace.wo_a_activation_e4m3 = 0;
  EXPECT_FALSE(DeepSeekAttentionProjectionSubmissionAssembler::Assemble(
                   weights(), invalid_workspace, 8, 4, 0x301, 0x302, 0x303,
                   104, 0x8)
                   .ok());
}

}  // namespace
}  // namespace pih
