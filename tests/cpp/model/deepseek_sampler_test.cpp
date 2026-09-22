#include "pih/model/deepseek_sampler.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>

namespace pih {
namespace {

TEST(DeepSeekSamplerTest, MatchesPhilox4x32TenZeroVectorByOrdinal) {
  EXPECT_EQ(deepseek_philox_word(0, 0).value(), 0x6627e8d5U);
  EXPECT_EQ(deepseek_philox_word(0, 1).value(), 0xe169c58dU);
  EXPECT_EQ(deepseek_philox_word(0, 2).value(), 0xbc57ac4cU);
  EXPECT_EQ(deepseek_philox_word(0, 3).value(), 0x9b00dbd8U);
}

TEST(DeepSeekSamplerTest, GreedyBreaksFiniteLogitTiesByLowestTokenId) {
  const std::array<float, 4> logits{-1.0F, 3.0F, 3.0F, 2.0F};
  auto sampled = deepseek_greedy_sample(logits);
  ASSERT_TRUE(sampled.ok()) << sampled.status().message();
  EXPECT_EQ(*sampled, 1U);
}

TEST(DeepSeekSamplerTest, GreedyRejectsAnyNonFiniteLogit) {
  const std::array<float, 3> logits{1.0F,
                                    std::numeric_limits<float>::infinity(),
                                    2.0F};
  EXPECT_EQ(deepseek_greedy_sample(logits).status().code(),
            StatusCode::kInternal);
}

TEST(DeepSeekSamplerTest, MinTokensMaskPrecedesGreedyAndStochasticLogprobs) {
  const std::array<float, 3> logits{9.0F, 8.0F, 7.0F};
  DeepSeekSuppressedTokenSet suppressed;
  suppressed.token_ids[0] = 0;
  suppressed.token_ids[1] = 1;
  suppressed.token_count = 2;
  auto greedy = deepseek_greedy_sample(logits, suppressed);
  ASSERT_TRUE(greedy.ok()) << greedy.status().message();
  EXPECT_EQ(*greedy, 2U);

  DeepSeekSamplingDescriptor descriptor;
  descriptor.temperature = 1.0F;
  descriptor.top_p = 1.0F;
  descriptor.top_k = 1;
  descriptor.top_logprobs_count = 1;
  descriptor.suppressed_tokens = suppressed;
  auto sampled = deepseek_sample_cpu(logits, descriptor);
  ASSERT_TRUE(sampled.ok()) << sampled.status().message();
  EXPECT_EQ(sampled->token_id, 2U);
  EXPECT_FLOAT_EQ(sampled->selected_logprob, 0.0F);
  EXPECT_EQ(sampled->top_token_ids, std::vector<std::uint32_t>({2}));
  EXPECT_EQ(sampled->top_logprobs, std::vector<float>({0.0F}));
}

TEST(DeepSeekSamplerTest, RejectsNonCanonicalSuppressedTokenSet) {
  const std::array<float, 3> logits{9.0F, 8.0F, 7.0F};
  DeepSeekSuppressedTokenSet duplicate;
  duplicate.token_ids[0] = 0;
  duplicate.token_ids[1] = 0;
  duplicate.token_count = 2;
  EXPECT_FALSE(deepseek_greedy_sample(logits, duplicate).ok());

  DeepSeekSamplingDescriptor descriptor;
  descriptor.suppressed_tokens.token_ids[1] = 1;
  descriptor.suppressed_tokens.token_count = 1;
  EXPECT_FALSE(deepseek_sample_cpu(logits, descriptor).ok());
}

TEST(DeepSeekSamplerTest,
     StochasticTopKUsesFullDistributionSelectedLogprob) {
  const std::array<float, 2> logits{1.0F, 2.0F};
  DeepSeekSamplingDescriptor descriptor;
  descriptor.temperature = 1.0F;
  descriptor.top_p = 1.0F;
  descriptor.top_k = 1;
  descriptor.seed = 0;
  descriptor.sample_ordinal = 0;
  auto sampled = deepseek_sample_cpu(logits, descriptor);
  ASSERT_TRUE(sampled.ok()) << sampled.status().message();
  EXPECT_EQ(sampled->token_id, 1U);
  EXPECT_NEAR(sampled->selected_logprob, -0.31326166F, 1.0e-6F);
  EXPECT_EQ(sampled->rng_word, 0x6627e8d5U);
}

}  // namespace
}  // namespace pih
