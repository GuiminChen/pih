#include "pih/backend/cuda/deepseek_endpoint.h"
#include "pih/model/deepseek_endpoint_oracle.h"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace pih { namespace {

TEST(DeepSeekEndpointTest, EmbeddingExpandsEachTokenToFourIdenticalStreams) {
  const std::vector<std::uint32_t> ids{2, 0};
  const std::vector<BFloat16> weight{
      BFloat16::FromFloat(1), BFloat16::FromFloat(2),
      BFloat16::FromFloat(3), BFloat16::FromFloat(4),
      BFloat16::FromFloat(5), BFloat16::FromFloat(6)};
  std::vector<BFloat16> output(2U * 4U * 2U);
  ASSERT_TRUE(deepseek_embedding_expand_oracle(ids, weight, 3, 2, output).ok());
  for (std::uint32_t stream = 0; stream < 4; ++stream) {
    EXPECT_FLOAT_EQ(output[stream * 2].to_float(), 5.0F);
    EXPECT_FLOAT_EQ(output[stream * 2 + 1].to_float(), 6.0F);
  }
  EXPECT_FLOAT_EQ(output[8].to_float(), 1.0F);
}

TEST(DeepSeekEndpointTest, HcHeadUsesFlattenedRmsAndSigmoidGates) {
  std::vector<BFloat16> input(4U * 2U);
  for (std::uint32_t stream = 0; stream < 4; ++stream) {
    input[stream * 2] = BFloat16::FromFloat(static_cast<float>(stream + 1));
    input[stream * 2 + 1] = BFloat16::FromFloat(0.0F);
  }
  std::vector<float> fn(4U * 8U, 0.0F);
  std::vector<float> scale{1.0F};
  std::vector<float> base(4, 0.0F);
  std::vector<BFloat16> output(2);
  ASSERT_TRUE(deepseek_hc_head_oracle(input, fn, scale, base, 1, 2,
                                      1.0e-6F, 1.0e-6F, output).ok());
  const float gate = 0.5F + 1.0e-6F;
  EXPECT_EQ(output[0], BFloat16::FromFloat(gate * 10.0F));
  EXPECT_EQ(output[1], BFloat16::FromFloat(0.0F));
}

TEST(DeepSeekEndpointTest, LmHeadAccumulatesBf16OperandsInFp32) {
  const std::vector<BFloat16> input{BFloat16::FromFloat(2),
                                    BFloat16::FromFloat(-1)};
  const std::vector<BFloat16> weight{
      BFloat16::FromFloat(3), BFloat16::FromFloat(4),
      BFloat16::FromFloat(-2), BFloat16::FromFloat(5)};
  std::vector<float> logits(2);
  ASSERT_TRUE(deepseek_lm_head_oracle(input, weight, 1, 2, 2, logits).ok());
  EXPECT_FLOAT_EQ(logits[0], 2.0F);
  EXPECT_FLOAT_EQ(logits[1], -9.0F);
}

TEST(DeepSeekEndpointTest, OracleFailureDoesNotPublishPartialOutput) {
  std::vector<BFloat16> output(8, BFloat16::FromFloat(9));
  const std::vector<std::uint32_t> ids{3};
  const std::vector<BFloat16> weight(3U * 2U);
  EXPECT_FALSE(deepseek_embedding_expand_oracle(ids, weight, 3, 2,
                                                 output).ok());
  for (const auto value : output) EXPECT_EQ(value, BFloat16::FromFloat(9));
}

TEST(DeepSeekEndpointTest, NonfiniteHeadInputDoesNotPublishOutput) {
  std::vector<BFloat16> input(8, BFloat16::FromFloat(1));
  std::vector<float> fn(32, 0.0F);
  fn[31] = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> scale{1.0F};
  const std::vector<float> base(4, 0.0F);
  std::vector<BFloat16> output(2, BFloat16::FromFloat(7));
  EXPECT_FALSE(deepseek_hc_head_oracle(input, fn, scale, base, 1, 2,
                                       1.0e-6F, 1.0e-6F, output).ok());
  EXPECT_EQ(output[0], BFloat16::FromFloat(7));
  EXPECT_EQ(output[1], BFloat16::FromFloat(7));
}

TEST(DeepSeekEndpointTest, CudaLaunchesPinPublishedModelGeometry) {
  DeepSeekEmbeddingLaunch embedding{1, 2, 3, 4, 5, 2, 129280, 4096, 4};
  EXPECT_TRUE(validate_deepseek_embedding_launch(embedding).ok());
  embedding.vocab_size = 129279;
  EXPECT_FALSE(validate_deepseek_embedding_launch(embedding).ok());
  embedding.vocab_size = 129280;
  embedding.output_hc_bf16 = embedding.weight_bf16;
  EXPECT_FALSE(validate_deepseek_embedding_launch(embedding).ok());

  DeepSeekHcHeadLaunch head{1, 2, 3, 4, 5, 6, 7, 2, 4096, 4,
                             1.0e-6F, 1.0e-6F};
  EXPECT_TRUE(validate_deepseek_hc_head_launch(head).ok());
  head.hc_multiplicity = 1;
  EXPECT_FALSE(validate_deepseek_hc_head_launch(head).ok());

  DeepSeekLmHeadLaunch logits{1, 2, 3, 4, 5, 2, 129280, 4096};
  EXPECT_TRUE(validate_deepseek_lm_head_launch(logits).ok());
  logits.output_rows = 0;
  EXPECT_FALSE(validate_deepseek_lm_head_launch(logits).ok());

  DeepSeekDsparkMarkovLaunch markov{1, 2, 3, 4, 5, 6, 7, 8,
                                    5, 129280, 256};
  EXPECT_TRUE(validate_deepseek_dspark_markov_launch(markov).ok());
  markov.row_count = 6;
  EXPECT_FALSE(validate_deepseek_dspark_markov_launch(markov).ok());

  DeepSeekDsparkConfidenceLaunch confidence{1, 2, 3, 4, 5, 6,
                                             5, 4096, 256};
  EXPECT_TRUE(validate_deepseek_dspark_confidence_launch(confidence).ok());
  confidence.markov_rank = 255;
  EXPECT_FALSE(validate_deepseek_dspark_confidence_launch(confidence).ok());
}

TEST(DeepSeekEndpointTest, DSparkMarkovAddsTokenConditionedBiasInFp32) {
  const std::vector<std::uint32_t> ids{1, 0};
  const std::vector<BFloat16> embedding{
      BFloat16::FromFloat(1), BFloat16::FromFloat(2),
      BFloat16::FromFloat(3), BFloat16::FromFloat(4),
      BFloat16::FromFloat(5), BFloat16::FromFloat(6)};
  const std::vector<BFloat16> head{
      BFloat16::FromFloat(1), BFloat16::FromFloat(0),
      BFloat16::FromFloat(0), BFloat16::FromFloat(1),
      BFloat16::FromFloat(1), BFloat16::FromFloat(-1)};
  const std::vector<float> raw{10, 20, 30, 40, 50, 60};
  std::vector<BFloat16> embeds(4);
  std::vector<float> biased(6);
  ASSERT_TRUE(deepseek_dspark_markov_oracle(
      ids, embedding, head, raw, 3, 2, embeds, biased).ok());
  EXPECT_EQ(embeds[0], BFloat16::FromFloat(3));
  EXPECT_FLOAT_EQ(biased[0], 13.0F);
  EXPECT_FLOAT_EQ(biased[1], 24.0F);
  EXPECT_FLOAT_EQ(biased[2], 29.0F);
  EXPECT_FLOAT_EQ(biased[3], 41.0F);
  EXPECT_FLOAT_EQ(biased[4], 52.0F);
  EXPECT_FLOAT_EQ(biased[5], 59.0F);
}

TEST(DeepSeekEndpointTest, DSparkConfidenceConcatenatesHiddenAndMarkov) {
  const std::vector<BFloat16> hidden{
      BFloat16::FromFloat(1), BFloat16::FromFloat(2)};
  const std::vector<BFloat16> markov{BFloat16::FromFloat(3)};
  const std::vector<float> weight{2, -1, 0.5F};
  std::vector<float> confidence(1);
  ASSERT_TRUE(deepseek_dspark_confidence_oracle(
      hidden, markov, weight, 1, 2, 1, confidence).ok());
  EXPECT_FLOAT_EQ(confidence[0], 1.5F);
}

TEST(DeepSeekEndpointTest, DSparkArgmaxUsesLowestIdForExactTies) {
  std::vector<float> logits(129280, -3.0F);
  logits[19] = 7.0F;
  logits[23] = 7.0F;
  ASSERT_TRUE(deepseek_argmax_oracle(logits).ok());
  EXPECT_EQ(*deepseek_argmax_oracle(logits), 19U);
  logits[31] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(deepseek_argmax_oracle(logits).ok());

  DeepSeekArgmaxLaunch launch{1, 2, 3, 4, 129280};
  EXPECT_TRUE(validate_deepseek_argmax_launch(launch).ok());
  launch.suppressed_tokens.token_ids[0] = 1;
  launch.suppressed_tokens.token_ids[1] = 17;
  launch.suppressed_tokens.token_count = 2;
  EXPECT_TRUE(validate_deepseek_argmax_launch(launch).ok());
  launch.suppressed_tokens.token_ids[1] = 1;
  EXPECT_FALSE(validate_deepseek_argmax_launch(launch).ok());
  launch.suppressed_tokens.token_ids[1] = 17;
  launch.vocab_size = 129279;
  EXPECT_FALSE(validate_deepseek_argmax_launch(launch).ok());

  DeepSeekStochasticSampleLaunch stochastic{
      1, 2, 3, 4, 5, 6, 7, 8,
      129280, 129280, 0.8F, 0.9F, 64, 11, 12};
  EXPECT_TRUE(validate_deepseek_stochastic_sample_launch(stochastic).ok());
  stochastic.suppressed_tokens.token_ids[0] = 1;
  stochastic.suppressed_tokens.token_count = 1;
  EXPECT_TRUE(validate_deepseek_stochastic_sample_launch(stochastic).ok());
  stochastic.suppressed_tokens.token_ids[1] = 17;
  EXPECT_FALSE(validate_deepseek_stochastic_sample_launch(stochastic).ok());
  stochastic.suppressed_tokens.token_ids[1] = 0;
  stochastic.workspace_capacity = 129279;
  EXPECT_FALSE(validate_deepseek_stochastic_sample_launch(stochastic).ok());
  stochastic.workspace_capacity = 129280;
  stochastic.temperature = 0.0F;
  EXPECT_FALSE(validate_deepseek_stochastic_sample_launch(stochastic).ok());
}

TEST(DeepSeekEndpointTest, DSparkDraftInitUsesInputThenFourNoiseTokens) {
  const std::vector<std::uint32_t> input{2};
  const std::vector<BFloat16> embedding{
      BFloat16::FromFloat(1), BFloat16::FromFloat(2),
      BFloat16::FromFloat(3), BFloat16::FromFloat(4),
      BFloat16::FromFloat(5), BFloat16::FromFloat(6)};
  std::vector<std::uint32_t> ids(5);
  std::vector<BFloat16> hc(5U * 4U * 2U);
  ASSERT_TRUE(deepseek_dspark_draft_init_oracle(
      input, embedding, 1, 5, 3, 2, ids, hc).ok());
  EXPECT_EQ(ids, std::vector<std::uint32_t>({2, 1, 1, 1, 1}));
  for (std::uint32_t stream = 0; stream < 4; ++stream) {
    EXPECT_EQ(hc[stream * 2], BFloat16::FromFloat(5));
    EXPECT_EQ(hc[(4 + stream) * 2], BFloat16::FromFloat(3));
  }
}

TEST(DeepSeekEndpointTest, DSparkDraftInitPinsPublishedGeometry) {
  DeepSeekDsparkDraftInitLaunch launch{
      1, 2, 3, 4, 5, 6, 2, 17, 5, 129280, 4096, 4};
  EXPECT_TRUE(validate_deepseek_dspark_draft_init_launch(launch).ok());
  launch.block_size = 7;
  EXPECT_FALSE(validate_deepseek_dspark_draft_init_launch(launch).ok());
  launch.block_size = 5;
  launch.noise_token_id = 129280;
  EXPECT_FALSE(validate_deepseek_dspark_draft_init_launch(launch).ok());
}

}}  // namespace pih::<anonymous>
