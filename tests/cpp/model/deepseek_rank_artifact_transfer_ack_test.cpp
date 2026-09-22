#include "pih/model/deepseek_rank_artifact_transfer_ack.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest digest(std::uint8_t seed) {
  Sha256Digest value{};
  for (std::size_t index = 0; index < value.bytes.size(); ++index) {
    value.bytes[index] = static_cast<std::byte>(seed + index);
  }
  return value;
}

DeepSeekRankArtifactTransferAckFields valid_fields() {
  return {7,
          8,
          4,
          2,
          1,
          3,
          16,
          16,
          32,
          false,
          10,
          100,
          200,
          300,
          400,
          digest(1),
          digest(2),
          digest(3),
          digest(4),
          digest(5)};
}

TEST(DeepSeekRankArtifactTransferAckTest,
     RoundTripsExactFixedFrameAndRejectsEveryBitRegionMutation) {
  auto ack = DeepSeekRankArtifactTransferAck::Create(valid_fields());
  ASSERT_TRUE(ack.ok()) << ack.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactTransferAckAbi,
            "pih_deepseek_rank_artifact_transfer_ack_v1");
  EXPECT_EQ(kDeepSeekRankArtifactTransferAckFrameAbi,
            "pih_deepseek_rank_artifact_transfer_ack_frame_v1");
  EXPECT_EQ(kDeepSeekRankArtifactTransferAckBytes, 288U);
  const auto frame = encode_deepseek_rank_artifact_transfer_ack(*ack);
  auto decoded = decode_deepseek_rank_artifact_transfer_ack(frame);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->fields().batch_index, 1U);
  EXPECT_EQ(decoded->fields().cumulative_descriptor_count, 32U);
  EXPECT_EQ(decoded->ack_root(), ack->ack_root());

  for (std::size_t offset = 0; offset < frame.size(); ++offset) {
    auto changed = frame;
    changed[offset] ^= std::byte{0x01};
    EXPECT_FALSE(decode_deepseek_rank_artifact_transfer_ack(changed).ok())
        << "offset=" << offset;
  }
  std::vector<std::byte> trailing(frame.begin(), frame.end());
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(decode_deepseek_rank_artifact_transfer_ack(trailing).ok());
}

TEST(DeepSeekRankArtifactTransferAckTest,
     EnforcesCanonicalBatchGeometryAndFinalFlag) {
  auto fields = valid_fields();
  fields.first_descriptor_ordinal = 15;
  EXPECT_FALSE(DeepSeekRankArtifactTransferAck::Create(fields).ok());
  fields = valid_fields();
  fields.descriptor_count = 15;
  fields.cumulative_descriptor_count = 31;
  EXPECT_FALSE(DeepSeekRankArtifactTransferAck::Create(fields).ok());
  fields = valid_fields();
  fields.final_batch = true;
  EXPECT_FALSE(DeepSeekRankArtifactTransferAck::Create(fields).ok());

  fields = valid_fields();
  fields.batch_index = 2;
  fields.first_descriptor_ordinal = 32;
  fields.descriptor_count = 3;
  fields.cumulative_descriptor_count = 35;
  fields.final_batch = true;
  EXPECT_TRUE(DeepSeekRankArtifactTransferAck::Create(fields).ok());
}

}  // namespace
}  // namespace pih
