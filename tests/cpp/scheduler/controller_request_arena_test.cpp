#include "pih/scheduler/controller_request_arena.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {

TEST(ControllerRequestArenaTest, PublishesClaimsAndReusesFixedBacking) {
  auto arena = ControllerRequestArena::Create({1, 4, 8}).value();
  const std::array<std::uint32_t, 3> tokens{1, 2, 3};
  auto receipt = arena.publish(11, tokens, 2);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(arena.published_count(), 1U);
  auto view = arena.claim(11, receipt->payload_digest);
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view->prompt_token_ids[2], 3U);
  EXPECT_EQ(view->maximum_new_tokens, 2U);
  const auto address = view->prompt_token_ids.data();
  const auto old_second_token_address = address + 1;
  EXPECT_TRUE(arena.release_claimed(view->slot_index, view->slot_generation).ok());
  EXPECT_EQ(*old_second_token_address, 0U);
  const std::array<std::uint32_t, 1> next_tokens{9};
  auto next_receipt = arena.publish(12, next_tokens, 1).value();
  auto next = arena.claim(12, next_receipt.payload_digest).value();
  EXPECT_EQ(next.prompt_token_ids.data(), address);
  EXPECT_EQ(next.slot_generation, view->slot_generation + 1);
}

TEST(ControllerRequestArenaTest, DigestMismatchDoesNotClaimPublishedPayload) {
  auto arena = ControllerRequestArena::Create({1, 2, 4}).value();
  const std::array<std::uint32_t, 1> tokens{4};
  auto receipt = arena.publish(1, tokens, 1).value();
  const auto wrong = controller_request_payload_digest(2, tokens, 1).value();
  EXPECT_FALSE(arena.claim(1, wrong).ok());
  EXPECT_EQ(arena.published_count(), 1U);
  EXPECT_EQ(arena.claim(1, receipt.payload_digest)->prompt_token_ids[0], 4U);
}

TEST(ControllerRequestArenaTest, RollbackIsGenerationCheckedAndCreditBounded) {
  auto arena = ControllerRequestArena::Create({1, 2, 4}).value();
  const std::array<std::uint32_t, 1> tokens{4};
  auto receipt = arena.publish(1, tokens, 1).value();
  EXPECT_FALSE(arena.publish(2, tokens, 1).ok());
  auto stale = receipt;
  ++stale.slot_generation;
  EXPECT_FALSE(arena.rollback_published(stale).ok());
  EXPECT_TRUE(arena.rollback_published(receipt).ok());
  EXPECT_EQ(arena.published_count(), 0U);
  EXPECT_TRUE(arena.publish(2, tokens, 1).ok());
}

TEST(ControllerRequestArenaTest, RejectsVocabularyAndContextOverflowBeforeWrite) {
  auto arena = ControllerRequestArena::Create({1, 3, 4}).value();
  const std::array<std::uint32_t, 2> invalid_vocab{1, 151936};
  EXPECT_FALSE(arena.publish(1, invalid_vocab, 1).ok());
  const std::array<std::uint32_t, 3> tokens{1, 2, 3};
  EXPECT_FALSE(arena.publish(1, tokens, 2).ok());
  EXPECT_EQ(arena.published_count(), 0U);
}

TEST(ControllerRequestArenaTest,
     SamplingDescriptorIsValidatedDigestedAndOwnedByRequestSlot) {
  auto arena = ControllerRequestArena::Create({1, 4, 8}).value();
  const std::array<std::uint32_t, 2> tokens{1, 2};
  ControllerRequestSampling sampling;
  sampling.mode = ControllerSamplingMode::kStochastic;
  sampling.temperature = 0.75F;
  sampling.top_p = 0.9F;
  sampling.top_k = 7;
  sampling.effective_seed = 99;
  sampling.minimum_new_tokens = 2;
  sampling.logprobs_enabled = true;
  sampling.top_logprobs_count = 5;
  sampling.stop_token_count = 2;
  sampling.stop_token_ids[0] = 17;
  sampling.stop_token_ids[1] = 23;
  auto receipt = arena.publish(11, tokens, 4, sampling);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  sampling.effective_seed = 100;
  auto view = arena.claim(11, receipt->payload_digest);
  ASSERT_TRUE(view.ok()) << view.status().message();
  EXPECT_EQ(view->sampling.effective_seed, 99U);
  EXPECT_EQ(view->sampling.sample_ordinal, 0U);
  EXPECT_EQ(view->sampling.minimum_new_tokens, 2U);
  EXPECT_EQ(view->sampling.top_logprobs_count, 5U);
  EXPECT_EQ(view->sampling.stop_token_count, 2U);
  EXPECT_EQ(view->sampling.stop_token_ids[0], 17U);
  EXPECT_EQ(view->sampling.stop_token_ids[1], 23U);
  EXPECT_NE(controller_request_payload_digest(11, tokens, 4, sampling).value(),
            receipt->payload_digest);

  ControllerRequestSampling invalid;
  invalid.mode = ControllerSamplingMode::kGreedy;
  invalid.temperature = 1.0F;
  EXPECT_EQ(arena.publish(12, tokens, 4, invalid).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(ControllerRequestArenaTest,
     RejectsNonCanonicalOrOutOfVocabularyStopTokenSets) {
  auto arena = ControllerRequestArena::Create({1, 4, 8}).value();
  const std::array<std::uint32_t, 1> tokens{1};
  ControllerRequestSampling duplicate;
  duplicate.stop_token_count = 2;
  duplicate.stop_token_ids[0] = 7;
  duplicate.stop_token_ids[1] = 7;
  EXPECT_EQ(arena.publish(1, tokens, 2, duplicate).status().code(),
            StatusCode::kInvalidArgument);
  ControllerRequestSampling invalid_vocab;
  invalid_vocab.stop_token_count = 1;
  invalid_vocab.stop_token_ids[0] = 151936;
  EXPECT_EQ(arena.publish(2, tokens, 2, invalid_vocab).status().code(),
            StatusCode::kInvalidArgument);
}

}  // namespace pih
