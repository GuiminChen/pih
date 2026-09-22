#include "pih/model/deepseek_endpoint_plan_input_assembler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekEndpointWeightBindings endpoint_weights() {
  return {0x10000, 0x20000, 0x30000, 0x40000,
          0x50000, 0x60000, 7};
}

TEST(DeepSeekEndpointPlanInputAssemblerTest, AssemblesWorldOneEndpointDataflow) {
  DeepSeekStagePlan stage{0, {0, 42}, true, true, true};
  const DeepSeekEndpointDeviceView device{
      0x70000, 0x80000, 0x90000, 0xA0000, 0xA1000, 0xB0000,
      0xA2000};
  auto work = DeepSeekEndpointPlanInputAssembler::Assemble(
      stage, 3, 0xC0000, 0xD0000, endpoint_weights(), device,
      reinterpret_cast<DeepSeekEndpointSequenceExecutor*>(0xE0000), 0xF0000,
      8);
  ASSERT_TRUE(work.ok()) << work.status().message();
  EXPECT_EQ(work->executor,
            reinterpret_cast<DeepSeekEndpointSequenceExecutor*>(0xE0000));
  EXPECT_EQ(work->embedding.token_ids_u32, 0xC0000U);
  EXPECT_EQ(work->embedding.weight_bf16, 0x10000U);
  EXPECT_EQ(work->embedding.output_hc_bf16, 0x70000U);
  EXPECT_EQ(work->embedding.token_count, 3U);
  EXPECT_EQ(work->head.hc.input_hc_bf16, 0xD0000U);
  EXPECT_EQ(work->head.hc.output_bf16, 0x80000U);
  EXPECT_EQ(work->head.rms.input_bf16, 0x80000U);
  EXPECT_EQ(work->head.rms.output_bf16, 0x90000U);
  EXPECT_EQ(work->head.lm.input_bf16, 0x90000U);
  EXPECT_EQ(work->head.lm.logits_f32, 0xA0000U);
  EXPECT_EQ(work->head.sample.token_id_u32, 0xA1000U);
  EXPECT_EQ(work->head.sample.selected_logprob_f32, 0xA2000U);
}

TEST(DeepSeekEndpointPlanInputAssemblerTest,
     RejectsMissingStageOwnedInputBeforePublication) {
  DeepSeekStagePlan last{1, {20, 42}, false, true, true};
  auto weights = endpoint_weights();
  weights.embedding_weight_bf16 = 0;
  const DeepSeekEndpointDeviceView device{
      0, 0x80000, 0x90000, 0xA0000, 0xA1000, 0xB0000};
  EXPECT_FALSE(DeepSeekEndpointPlanInputAssembler::Assemble(
      last, 1, 0, 0, weights, device,
      reinterpret_cast<DeepSeekEndpointSequenceExecutor*>(0xE0000), 0xF0000,
      8).ok());
}

TEST(DeepSeekEndpointPlanInputAssemblerTest,
     RejectsMainGreedyHeadWithoutLogprobStorage) {
  DeepSeekStagePlan last{1, {20, 42}, false, true, true};
  const DeepSeekEndpointDeviceView device{
      0, 0x80000, 0x90000, 0xA0000, 0xA1000, 0xB0000};
  EXPECT_FALSE(DeepSeekEndpointPlanInputAssembler::Assemble(
      last, 1, 0, 0xD0000, endpoint_weights(), device,
      reinterpret_cast<DeepSeekEndpointSequenceExecutor*>(0xE0000), 0xF0000,
      8).ok());
}

TEST(DeepSeekEndpointPlanInputAssemblerTest,
     AssemblesStochasticSamplerFromPreparedRequestDescriptor) {
  DeepSeekStagePlan last{1, {20, 42}, false, true, true};
  const DeepSeekEndpointDeviceView device{
      0, 0x80000, 0x90000, 0xA0000, 0xA1000, 0xB0000,
      0xA2000, 0xA3000, 0xA4000, 0xA5000, 0xA6000, 0xA7000};
  const DeepSeekPreparedSamplingInput sampling{
      17, DeepSeekSamplingMode::kStochastic,
      {0.8F, 0.9F, 64, 7, 3, true, 3}};
  auto work = DeepSeekEndpointPlanInputAssembler::Assemble(
      last, 1, 0, 0xD0000, endpoint_weights(), device,
      reinterpret_cast<DeepSeekEndpointSequenceExecutor*>(0xE0000), 0xF0000,
      8, sampling);
  ASSERT_TRUE(work.ok()) << work.status().message();
  ASSERT_TRUE(work->head.stochastic_sample.has_value());
  EXPECT_EQ(work->head.stochastic_sample->workspace_values_f32, 0xA4000U);
  EXPECT_EQ(work->head.stochastic_sample->workspace_ids_u32, 0xA5000U);
  EXPECT_EQ(work->head.stochastic_sample->sample_ordinal, 3U);
  EXPECT_EQ(work->head.stochastic_sample->top_logprobs_ids_u32, 0xA6000U);
  EXPECT_EQ(work->head.stochastic_sample->top_logprobs_f32, 0xA7000U);
  EXPECT_EQ(work->head.stochastic_sample->top_logprobs_count, 3U);
  EXPECT_EQ(work->head.sample.logits_f32, 0U);
}

TEST(DeepSeekEndpointPlanInputAssemblerTest,
     BindsGreedyTopLogprobsWithoutStochasticRng) {
  DeepSeekStagePlan last{1, {20, 42}, false, true, true};
  const DeepSeekEndpointDeviceView device{
      0, 0x80000, 0x90000, 0xA0000, 0xA1000, 0xB0000,
      0xA2000, 0xA3000, 0xA4000, 0xA5000, 0xA6000, 0xA7000};
  DeepSeekPreparedSamplingInput sampling{
      17, DeepSeekSamplingMode::kGreedy,
      {0.0F, 1.0F, std::nullopt, 0, 0, true, 3}};
  sampling.descriptor.suppressed_tokens.token_ids[0] = 1;
  sampling.descriptor.suppressed_tokens.token_ids[1] = 17;
  sampling.descriptor.suppressed_tokens.token_count = 2;
  auto work = DeepSeekEndpointPlanInputAssembler::Assemble(
      last, 1, 0, 0xD0000, endpoint_weights(), device,
      reinterpret_cast<DeepSeekEndpointSequenceExecutor*>(0xE0000), 0xF0000,
      8, sampling);
  ASSERT_TRUE(work.ok()) << work.status().message();
  EXPECT_FALSE(work->head.stochastic_sample.has_value());
  EXPECT_EQ(work->head.sample.top_logprobs_ids_u32, 0xA6000U);
  EXPECT_EQ(work->head.sample.top_logprobs_f32, 0xA7000U);
  EXPECT_EQ(work->head.sample.top_logprobs_count, 3U);
  EXPECT_EQ(work->head.sample.suppressed_tokens.token_count, 2U);
  EXPECT_EQ(work->head.sample.suppressed_tokens.token_ids[0], 1U);
  EXPECT_EQ(work->head.sample.suppressed_tokens.token_ids[1], 17U);
}

}  // namespace
}  // namespace pih
