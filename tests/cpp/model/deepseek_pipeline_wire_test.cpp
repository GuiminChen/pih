#include "pih/model/deepseek_pipeline_wire.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekPipelineWireMessage fixture() {
  DeepSeekPipelineWireMessage message;
  message.engine_epoch = 7;
  message.plan_sequence = 9;
  message.microbatch_id = 11;
  message.phase = DeepSeekPlanPhase::kDecode;
  message.records = {
      {{1, 2}, 3, 4, 0, 1, 8, 16, 8, 0, 5},
      {{6, 7}, 8, 9, 1, 1, 12, 20, 12, 0, 10},
  };
  message.token_ids = {101, 102};
  return message;
}

TEST(DeepSeekPipelineWireTest, EncodesFixedAbiAndRoundTrips) {
  auto encoded = DeepSeekPipelineWireCodec::Encode(fixture());
  ASSERT_TRUE(encoded.ok()) << encoded.status().message();
  EXPECT_EQ(encoded->size(), 256U);
  EXPECT_EQ((*encoded)[0], std::byte{1});
  EXPECT_EQ((*encoded)[1], std::byte{0});
  auto decoded = DeepSeekPipelineWireCodec::Decode(*encoded, 16, 4);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->engine_epoch, 7U);
  EXPECT_EQ(decoded->plan_sequence, 9U);
  EXPECT_EQ(decoded->records, fixture().records);
  EXPECT_EQ(decoded->token_ids, fixture().token_ids);
}

TEST(DeepSeekPipelineWireTest, MatchesPublishedCapacityExamples) {
  EXPECT_EQ(DeepSeekPipelineWireCodec::PaddedBytes(512, 32).value(), 4352U);
  EXPECT_EQ(DeepSeekPipelineWireCodec::PaddedBytes(1024, 32).value(), 6400U);
}

TEST(DeepSeekPipelineWireTest, RejectsHeaderPayloadAndReservedCorruption) {
  auto encoded = DeepSeekPipelineWireCodec::Encode(fixture());
  ASSERT_TRUE(encoded.ok());
  auto header_corrupt = *encoded;
  header_corrupt[8] ^= std::byte{1};
  EXPECT_FALSE(DeepSeekPipelineWireCodec::Decode(header_corrupt, 16, 4).ok());

  auto payload_corrupt = *encoded;
  payload_corrupt[DeepSeekPipelineWireCodec::kHeaderBytes + 3] ^= std::byte{1};
  EXPECT_FALSE(DeepSeekPipelineWireCodec::Decode(payload_corrupt, 16, 4).ok());

  auto reserved_corrupt = *encoded;
  reserved_corrupt[7] = std::byte{1};
  EXPECT_FALSE(DeepSeekPipelineWireCodec::Decode(reserved_corrupt, 16, 4).ok());
}

TEST(DeepSeekPipelineWireTest, RejectsLogicalSpanAndCapacityMismatch) {
  auto message = fixture();
  message.records[1].token_offset = 3;
  EXPECT_FALSE(DeepSeekPipelineWireCodec::Encode(message).ok());
  auto encoded = DeepSeekPipelineWireCodec::Encode(fixture());
  ASSERT_TRUE(encoded.ok());
  EXPECT_FALSE(DeepSeekPipelineWireCodec::Decode(*encoded, 1, 4).ok());
  EXPECT_FALSE(DeepSeekPipelineWireCodec::Decode(*encoded, 16, 1).ok());
}

TEST(DeepSeekPipelineWireTest, EncodesExplicitZeroTokenDrain) {
  auto message = fixture();
  message.phase = DeepSeekPlanPhase::kDrain;
  message.token_ids.clear();
  for (auto& record : message.records) {
    record.token_offset = 0;
    record.token_count = 0;
  }
  auto encoded = DeepSeekPipelineWireCodec::Encode(message);
  ASSERT_TRUE(encoded.ok()) << encoded.status().message();
  EXPECT_EQ((*encoded)[3], std::byte{1});
  auto decoded = DeepSeekPipelineWireCodec::Decode(*encoded, 16, 4);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_TRUE(decoded->token_ids.empty());
  EXPECT_EQ(decoded->phase, DeepSeekPlanPhase::kDrain);
}

}  // namespace
}  // namespace pih
