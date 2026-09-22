#include "pih/model/qwen3_sampler.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>

namespace pih { namespace {

TEST(Qwen3SamplerTest, UsesCanonicalPhiloxWordByAcceptedTokenOrdinal) {
  EXPECT_EQ(qwen3_philox_word(0, 0).value(), 0x6627e8d5U);
  EXPECT_EQ(qwen3_philox_word(0, 3).value(), 0x9b00dbd8U);
  EXPECT_EQ(qwen3_philox_word(0, 4).value(), 0xf8e4cca4U);
}

TEST(Qwen3SamplerTest, GreedyMasksEosBeforeTieBreakAndLogprob) {
  const std::array<float, 3> logits{9.0F, 8.0F, 8.0F};
  Qwen3SamplingDescriptor descriptor;
  descriptor.mode = Qwen3SamplingMode::kGreedy;
  descriptor.suppressed_token_count = 1;
  descriptor.suppressed_token_ids[0] = 0;
  descriptor.top_logprobs_count = 2;
  auto sampled = qwen3_sample_cpu(logits, descriptor);
  ASSERT_TRUE(sampled.ok()) << sampled.status().message();
  EXPECT_EQ(sampled->token_id, 1U);
  EXPECT_EQ(sampled->rng_word, 0U);
  EXPECT_EQ(sampled->top_token_ids,
            (std::vector<std::uint32_t>{1, 2}));
  EXPECT_NEAR(sampled->selected_logprob, -0.69314718F, 1.0e-6F);
}

TEST(Qwen3SamplerTest, GreedySuppressesEveryRequestLocalStopToken) {
  const std::array<float, 4> logits{9.0F, 8.0F, 7.0F, 6.0F};
  Qwen3SamplingDescriptor descriptor;
  descriptor.suppressed_token_count = 2;
  descriptor.suppressed_token_ids[0] = 0;
  descriptor.suppressed_token_ids[1] = 1;
  descriptor.top_logprobs_count = 2;
  auto sampled = qwen3_sample_cpu(logits, descriptor);
  ASSERT_TRUE(sampled.ok()) << sampled.status().message();
  EXPECT_EQ(sampled->token_id, 2U);
  EXPECT_EQ(sampled->top_token_ids,
            (std::vector<std::uint32_t>{2, 3}));
}

TEST(Qwen3SamplerTest, RejectsTopLogprobsBeyondAllowedVocabulary) {
  const std::array logits{9.0F, 8.0F, 7.0F};
  Qwen3SamplingDescriptor descriptor;
  descriptor.top_logprobs_count = 2;
  descriptor.suppressed_token_count = 2;
  descriptor.suppressed_token_ids[0] = 0;
  descriptor.suppressed_token_ids[1] = 1;
  EXPECT_FALSE(qwen3_sample_cpu(logits, descriptor).ok());
}

TEST(Qwen3SamplerTest,
     StochasticFiltersAfterFullDistributionLogprobAndIsReplayable) {
  const std::array<float, 3> logits{1.0F, 2.0F, 3.0F};
  Qwen3SamplingDescriptor descriptor;
  descriptor.mode = Qwen3SamplingMode::kStochastic;
  descriptor.temperature = 1.0F;
  descriptor.top_p = 1.0F;
  descriptor.top_k = 1;
  descriptor.seed = 0;
  descriptor.sample_ordinal = 0;
  descriptor.top_logprobs_count = 2;
  auto first = qwen3_sample_cpu(logits, descriptor);
  auto replay = qwen3_sample_cpu(logits, descriptor);
  ASSERT_TRUE(first.ok()) << first.status().message();
  ASSERT_TRUE(replay.ok()) << replay.status().message();
  EXPECT_EQ(first->token_id, 2U);
  EXPECT_EQ(first->rng_word, 0x6627e8d5U);
  EXPECT_EQ(*first, *replay);
  EXPECT_NEAR(first->selected_logprob, -0.40760595F, 1.0e-6F);
  EXPECT_EQ(first->top_token_ids,
            (std::vector<std::uint32_t>{2, 1}));
}

TEST(Qwen3SamplerTest, RejectsNonfiniteLogitsAndInvalidModeParameters) {
  const std::array<float, 2> nonfinite{
      1.0F, std::numeric_limits<float>::infinity()};
  EXPECT_EQ(qwen3_sample_cpu(nonfinite, {}).status().code(),
            StatusCode::kInternal);
  Qwen3SamplingDescriptor invalid;
  invalid.mode = Qwen3SamplingMode::kGreedy;
  invalid.top_p = 0.5F;
  EXPECT_EQ(qwen3_sample_cpu(std::array<float, 2>{1.0F, 2.0F}, invalid)
                .status().code(),
            StatusCode::kInvalidArgument);
}

} }  // namespace pih::<anonymous>
