#include "pih/model/deepseek_dspark_three_stage_reference.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace pih {
namespace {

TEST(DeepSeekDsparkThreeStageReferenceTest,
     FreezesThreeHiddenStagesAndFivePositionHeadGeometry) {
  const auto first = build_deepseek_dspark_three_stage_reference().value();
  const auto second = build_deepseek_dspark_three_stage_reference().value();
  constexpr auto kHiddenElements =
      DeepSeekDsparkThreeStageReferenceReceipt::kBlockSize *
      DeepSeekDsparkThreeStageReferenceReceipt::kHcStreams *
      DeepSeekDsparkThreeStageReferenceReceipt::kHiddenSize;
  constexpr auto kLogitElements =
      DeepSeekDsparkThreeStageReferenceReceipt::kBlockSize *
      DeepSeekDsparkThreeStageReferenceReceipt::kVocabularySize;
  for (std::uint32_t stage = 0; stage < kDeepSeekDsparkStageCount; ++stage) {
    EXPECT_EQ(first.stage_hidden_hc[stage].size(), kHiddenElements);
    EXPECT_EQ(first.stage_hidden_hc[stage], second.stage_hidden_hc[stage]);
    if (stage != 0) {
      EXPECT_NE(first.stage_hidden_hc[stage],
                first.stage_hidden_hc[stage - 1]);
    }
  }
  EXPECT_EQ(first.head_hidden.size(), 20U);
  EXPECT_EQ(first.raw_logits.size(), kLogitElements);
  EXPECT_EQ(first.markov_bias.size(), kLogitElements);
  EXPECT_EQ(first.biased_logits.size(), kLogitElements);
  EXPECT_EQ(first.confidence.size(), 5U);
  EXPECT_EQ(first.proposal_token_ids, second.proposal_token_ids);
  EXPECT_EQ(first.causal_token_chain, second.causal_token_chain);
  EXPECT_EQ(first.raw_logits, second.raw_logits);
  EXPECT_EQ(first.markov_bias, second.markov_bias);
  EXPECT_EQ(first.confidence, second.confidence);
}

TEST(DeepSeekDsparkThreeStageReferenceTest,
     FreezesHiddenLogitMarkovProposalAndConfidenceValues) {
  const auto receipt = build_deepseek_dspark_three_stage_reference().value();
  EXPECT_EQ(receipt.stage_hidden_hc[0][0].bits, 16152U);
  EXPECT_EQ(receipt.stage_hidden_hc[1][1].bits, 16281U);
  EXPECT_EQ(receipt.stage_hidden_hc[2][2].bits, 16391U);
  EXPECT_NEAR(receipt.raw_logits.front(), 0.19277954F, 1.0e-6F);
  EXPECT_NEAR(receipt.raw_logits.back(), 2.6596069F, 1.0e-6F);
  EXPECT_NEAR(receipt.markov_bias.front(), 0.012530565F, 1.0e-6F);
  EXPECT_NEAR(receipt.markov_bias.back(), 0.11245728F, 1.0e-6F);
  EXPECT_EQ(receipt.proposal_token_ids,
            (std::array<std::uint32_t, 5>{6, 6, 6, 6, 6}));
  EXPECT_EQ(receipt.causal_token_chain,
            (std::array<std::uint32_t, 6>{2, 6, 6, 6, 6, 6}));
  EXPECT_NEAR(receipt.confidence.front(), -0.28957033F, 1.0e-6F);
  EXPECT_NEAR(receipt.confidence.back(), -0.3375586F, 1.0e-6F);
}

}  // namespace
}  // namespace pih
