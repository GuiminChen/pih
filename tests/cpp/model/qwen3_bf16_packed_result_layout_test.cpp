#include "pih/model/qwen3_bf16_packed_result_layout.h"

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenBf16PackedResultLayoutTest, AlignsAndPublishesExactActivePrefix) {
  auto layout = QwenBf16PackedResultLayout::Create(7).value();
  EXPECT_EQ(layout.sampled_token_ids(), (QwenBf16ArenaSpan{0, 28}));
  EXPECT_EQ(layout.selected_logprobs().offset_bytes, 256U);
  EXPECT_EQ(layout.rng_words().offset_bytes, 512U);
  EXPECT_EQ(layout.top_logprob_token_ids().offset_bytes, 768U);
  EXPECT_EQ(layout.top_logprobs().offset_bytes, 1536U);
  EXPECT_EQ(layout.top_logprob_counts().offset_bytes, 2304U);
  EXPECT_EQ(layout.device_error().offset_bytes, 2560U);
  EXPECT_EQ(layout.total_bytes(), 2816U);
  std::vector<std::byte> backing(layout.total_bytes());
  ASSERT_TRUE(layout.initialize(backing).ok());
  const std::uint32_t tokens[]{3, 5, 8};
  const std::uint32_t error = 0;
  std::memcpy(backing.data(), tokens, sizeof(tokens));
  std::memcpy(backing.data() + layout.device_error().offset_bytes, &error,
              sizeof(error));
  auto parsed = layout.parse(backing, 3, true);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(*parsed, (std::vector<std::uint32_t>{3, 5, 8}));
}

TEST(QwenBf16PackedResultLayoutTest,
     PublishesValidatedSamplingReceiptsWithFixedTopTwentyStride) {
  auto layout = QwenBf16PackedResultLayout::Create(2).value();
  std::vector<std::byte> backing(layout.total_bytes());
  ASSERT_TRUE(layout.initialize(backing).ok());
  const std::uint32_t tokens[]{3, 5};
  const float selected[]{-0.25F, -0.5F};
  const std::uint32_t rng[]{17, 19};
  const std::uint32_t counts[]{2, 1};
  const std::uint32_t top_ids[]{3, 7, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                5};
  const float top_values[]{-0.25F, -1.0F, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           -0.5F};
  const std::uint32_t error = 0;
  std::memcpy(backing.data() + layout.sampled_token_ids().offset_bytes,
              tokens, sizeof(tokens));
  std::memcpy(backing.data() + layout.selected_logprobs().offset_bytes,
              selected, sizeof(selected));
  std::memcpy(backing.data() + layout.rng_words().offset_bytes, rng,
              sizeof(rng));
  std::memcpy(backing.data() + layout.top_logprob_counts().offset_bytes,
              counts, sizeof(counts));
  std::memcpy(backing.data() + layout.top_logprob_token_ids().offset_bytes,
              top_ids, sizeof(top_ids));
  std::memcpy(backing.data() + layout.top_logprobs().offset_bytes,
              top_values, sizeof(top_values));
  std::memcpy(backing.data() + layout.device_error().offset_bytes, &error, 4);
  auto receipts = layout.parse_sampling(backing, 2, true);
  ASSERT_TRUE(receipts.ok()) << receipts.status().message();
  ASSERT_EQ(receipts->size(), 2U);
  EXPECT_EQ((*receipts)[0].token_id, 3U);
  EXPECT_FLOAT_EQ((*receipts)[0].selected_logprob, -0.25F);
  EXPECT_EQ((*receipts)[0].rng_word, 17U);
  EXPECT_EQ((*receipts)[0].top_logprob_count, 2U);
  EXPECT_EQ((*receipts)[0].top_token_ids[1], 7U);
  EXPECT_FLOAT_EQ((*receipts)[1].top_logprobs[0], -0.5F);
}

TEST(QwenBf16PackedResultLayoutTest, FailsClosedOnAuthorizationErrorAndToken) {
  auto layout = QwenBf16PackedResultLayout::Create(2).value();
  std::vector<std::byte> backing(layout.total_bytes());
  ASSERT_TRUE(layout.initialize(backing).ok());
  std::uint32_t value = 0;
  std::memcpy(backing.data() + layout.device_error().offset_bytes, &value, 4);
  EXPECT_FALSE(layout.parse(backing, 1, false).ok());
  value = 16;
  std::memcpy(backing.data() + layout.device_error().offset_bytes, &value, 4);
  EXPECT_FALSE(layout.parse(backing, 1, true).ok());
  value = 0;
  std::memcpy(backing.data() + layout.device_error().offset_bytes, &value, 4);
  value = QwenBf16PackedResultLayout::kVocabularySize;
  std::memcpy(backing.data(), &value, 4);
  EXPECT_FALSE(layout.parse(backing, 1, true).ok());
}

TEST(QwenBf16PackedResultLayoutTest, RejectsInvalidCapacityAndBacking) {
  EXPECT_FALSE(QwenBf16PackedResultLayout::Create(0).ok());
  EXPECT_FALSE(QwenBf16PackedResultLayout::Create(4097).ok());
  auto layout = QwenBf16PackedResultLayout::Create(4096).value();
  EXPECT_GT(layout.device_error().offset_bytes, 16384U);
  std::vector<std::byte> short_backing(layout.total_bytes() - 1);
  EXPECT_FALSE(layout.initialize(short_backing).ok());
}

}  // namespace
}  // namespace pih
