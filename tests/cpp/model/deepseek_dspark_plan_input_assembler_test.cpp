#include "pih/model/deepseek_dspark_plan_input_assembler.h"

#include <gtest/gtest.h>

#include "deepseek_dspark_weight_test_fixture.h"

namespace pih { namespace {

TEST(DeepSeekDsparkPlanInputAssemblerTest, AssemblesOfficialSingleSequenceFlow) {
  DeepSeekStagePlan stage{0, {0, 42}, true, true, true};
  DeepSeekEndpointWeightBindings endpoint;
  endpoint.embedding_weight_bf16 = 10;
  endpoint.head_weight_bf16 = 11;
  endpoint.generation = 29;
  auto weights = test::deepseek_dspark_test_weights();
  DeepSeekDsparkDeviceView device{100,200,300,400,500,600,700,800,
                                  0x1000000,0x2000000,0x3000000,900,1000,
                                  0x4000000};
  auto work = DeepSeekDsparkPlanInputAssembler::Assemble(
      stage, 31, endpoint, weights, device,
      reinterpret_cast<DeepSeekDsparkEmbedCoordinator*>(40),
      reinterpret_cast<DeepSeekDsparkHeadExecutor*>(42),
      reinterpret_cast<DeepSeekAttentionSequenceTransaction*>(43), 44, 4096);
  ASSERT_TRUE(work.ok()) << work.status().message();
  EXPECT_EQ(work->embed.main_quant.input_bf16, device.target_hidden_bf16);
  EXPECT_EQ(work->embed.draft_init.input_token_ids_u32, 31U);
  EXPECT_EQ(work->head.hc.input_hc_bf16, 200U);
  EXPECT_EQ(work->head.lm.logits_f32, device.raw_logits_f32);
  EXPECT_EQ(work->head.markov[4].token_ids_u32,
            device.draft_token_ids_u32 + 4U * sizeof(std::uint32_t));
  EXPECT_EQ(work->head.argmax[4].token_id_u32,
            device.draft_token_ids_u32 + 5U * sizeof(std::uint32_t));
  EXPECT_EQ(work->head.confidence.projection_weight_bf16,
            weights.head().confidence_weight_bf16);
  auto unbound = DeepSeekDsparkPlanInputAssembler::Assemble(
      stage, 31, endpoint, weights, device,
      reinterpret_cast<DeepSeekDsparkEmbedCoordinator*>(40),
      reinterpret_cast<DeepSeekDsparkHeadExecutor*>(42), nullptr, 44, 4096);
  ASSERT_TRUE(unbound.ok()) << unbound.status().message();
  EXPECT_EQ(unbound->transaction, nullptr);
}

TEST(DeepSeekDsparkPlanInputAssemblerTest, RejectsForeignWeightGeneration) {
  DeepSeekStagePlan stage{0, {0, 42}, true, true, true};
  DeepSeekEndpointWeightBindings endpoint;
  endpoint.generation = 29;
  auto weights = test::deepseek_dspark_test_weights(30);
  EXPECT_FALSE(DeepSeekDsparkPlanInputAssembler::Assemble(
      stage, 2, endpoint, weights, {}, nullptr, nullptr, nullptr, 3, 1).ok());
}

TEST(DeepSeekDsparkPlanInputAssemblerTest,
     AssemblesPrefillProjectionWithoutDraftOrHead) {
  DeepSeekStagePlan stage{0, {0, 42}, true, true, true};
  auto weights = test::deepseek_dspark_test_weights();
  DeepSeekDsparkDeviceView device{100, 200, 300, 400, 500, 600, 700, 800,
                                  0x1000000, 0x2000000, 0x3000000, 900, 1000,
                                  0x4000000, 0x5000000};
  auto work = DeepSeekDsparkPlanInputAssembler::AssemblePrefill(
      stage, 17, weights, device,
      reinterpret_cast<DeepSeekDsparkEmbedCoordinator*>(40),
      reinterpret_cast<DeepSeekAttentionSequenceTransaction*>(43), 44, 64);
  ASSERT_TRUE(work.ok()) << work.status().message();
  EXPECT_EQ(work->kind,
            DeepSeekDsparkStageWorkKind::kPrefillStateInitialization);
  EXPECT_EQ(work->embed.main_quant.input_bf16,
            device.target_hidden_bf16);
  EXPECT_EQ(work->embed.main_quant.token_count, 17U);
  EXPECT_EQ(work->embed.main_proj.m, 17U);
  EXPECT_EQ(work->embed.main_norm.rows, 17U);
  EXPECT_EQ(work->embed.draft_init.input_token_ids_u32, 0U);
  EXPECT_EQ(work->head_executor, nullptr);
}

TEST(DeepSeekDsparkPlanInputAssemblerTest,
     RejectsInvalidDeferredLaunchBeforeCommit) {
  DeepSeekStagePlan stage{0, {0, 42}, true, true, true};
  DeepSeekEndpointWeightBindings endpoint;
  endpoint.embedding_weight_bf16 = 10;
  endpoint.head_weight_bf16 = 11;
  endpoint.generation = 29;
  auto weights = test::deepseek_dspark_test_weights();
  DeepSeekDsparkDeviceView device{100, 200, 300, 0, 500, 600, 700, 800,
                                  0x1000000, 0x2000000, 0x3000000, 900, 1000,
                                  0x4000000};
  auto work = DeepSeekDsparkPlanInputAssembler::Assemble(
      stage, 31, endpoint, weights, device,
      reinterpret_cast<DeepSeekDsparkEmbedCoordinator*>(40),
      reinterpret_cast<DeepSeekDsparkHeadExecutor*>(42), nullptr, 44, 4096);
  EXPECT_FALSE(work.ok());
}

} }  // namespace pih
