#include "pih/model/deepseek_projected_compressor_update_assembler.h"

#include <gtest/gtest.h>

namespace pih { namespace {

DeepSeekCompressorProjectionSlice projection_slice() {
  return {0x30000, 0x40000, 0x50000, 0x60000,
          0x70000, 0x80000, 0x90000};
}

DeepSeekCompressorWeightBindings ratio4_weights() {
  DeepSeekCompressorWeightBindings value;
  value.main_wkv_bf16 = 0x100000;
  value.main_wgate_bf16 = 0x180000;
  value.main_ape_f32 = 0x200000;
  value.main_norm_bf16 = 0x300000;
  value.indexer_wq_b_e4m3 = 0x400000;
  value.indexer_wq_b_scale_bits = 0x480000;
  value.indexer_weights_proj_bf16 = 0x500000;
  value.indexer_wkv_bf16 = 0x600000;
  value.indexer_wgate_bf16 = 0x680000;
  value.indexer_ape_f32 = 0x700000;
  value.indexer_norm_bf16 = 0x800000;
  value.generation = 19;
  return value;
}

TEST(DeepSeekProjectedCompressorUpdateAssemblerTest,
     AssemblesRatio4MainAndIndexerWithPositionSelectedApeRows) {
  auto result = DeepSeekProjectedCompressorUpdateAssembler::AssembleDecode(
      7, DeepSeekCompressedAttentionKind::kRatio4, 0x11000,
      ratio4_weights(), projection_slice(), 0xa0000, 0xb0000, 6);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->ratio, 4U);
  EXPECT_EQ(result->main_state.ape_row_f32, 0x202000U);
  EXPECT_EQ(result->index_state.ape_row_f32, 0x700800U);
  EXPECT_EQ(result->main_state.output_f32, 0x50000U);
  EXPECT_EQ(result->index_state.output_f32, 0x80000U);
  EXPECT_EQ(result->main_state.projection.kv_weight_bf16, 0x100000U);
  EXPECT_EQ(result->main_state.projection.gate_weight_bf16, 0x180000U);
  EXPECT_EQ(result->index_state.projection.kv_weight_bf16, 0x600000U);
  EXPECT_FALSE(result->has_completed_slot);
}

TEST(DeepSeekProjectedCompressorUpdateAssemblerTest,
     AssemblesRatio128MainWithoutIndexer) {
  auto weights = ratio4_weights();
  weights.main_wkv_bf16 = 0x120000;
  weights.main_ape_f32 = 0x220000;
  weights.indexer_wq_b_e4m3 = 0;
  weights.indexer_weights_proj_bf16 = 0;
  weights.indexer_wkv_bf16 = 0;
  weights.indexer_ape_f32 = 0;
  weights.indexer_norm_bf16 = 0;
  auto result = DeepSeekProjectedCompressorUpdateAssembler::AssembleDecode(
      11, DeepSeekCompressedAttentionKind::kRatio128, 0x11000, weights,
      projection_slice(), 0xa0000, 0xb0000, 129);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->ratio, 128U);
  EXPECT_EQ(result->main_state.ape_row_f32, 0x220800U);
  EXPECT_EQ(result->index_state.kv_projection_f32, 0U);
}

TEST(DeepSeekProjectedCompressorUpdateAssemblerTest,
     RejectsMissingRatio4IndexerOrInvalidPosition) {
  auto weights = ratio4_weights();
  weights.indexer_ape_f32 = 0;
  EXPECT_FALSE(DeepSeekProjectedCompressorUpdateAssembler::AssembleDecode(
      7, DeepSeekCompressedAttentionKind::kRatio4, 0x11000, weights,
      projection_slice(), 0xa0000, 0xb0000, 6).ok());
  EXPECT_FALSE(DeepSeekProjectedCompressorUpdateAssembler::AssembleDecode(
      7, DeepSeekCompressedAttentionKind::kRatio4, 0x11000,
      ratio4_weights(), projection_slice(), 0xa0000, 0xb0000, 1048576).ok());
}

} }  // namespace pih
