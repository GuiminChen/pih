#include "pih/model/deepseek_rank_artifact_metadata_chunk.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/core/sha256.h"

namespace pih {
namespace {

Sha256Digest chunk_digest(std::uint8_t seed) {
  Sha256Digest result{};
  for (std::size_t index = 0; index < result.bytes.size(); ++index) {
    result.bytes[index] = static_cast<std::byte>(seed + index);
  }
  return result;
}

std::vector<std::byte> chunk_payload(std::size_t size,
                                     std::uint8_t seed = 1) {
  std::vector<std::byte> result(size);
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::byte>(seed + index % 251U);
  }
  return result;
}

DeepSeekRankArtifactMetadataChunkFields chunk_fields(
    std::span<const std::byte> payload, std::uint64_t total_blob_bytes,
    std::uint32_t chunk_index, std::uint32_t chunk_count) {
  return {7,
          8,
          2,
          1,
          chunk_index,
          chunk_count,
          static_cast<std::uint64_t>(chunk_index) *
              kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes,
          static_cast<std::uint32_t>(payload.size()),
          total_blob_bytes,
          10,
          11,
          12,
          13,
          14,
          chunk_digest(1),
          chunk_digest(2),
          chunk_digest(3),
          chunk_digest(4),
          chunk_digest(5),
          sha256(payload).value()};
}

DeepSeekRankArtifactMetadataChunkAckFields ack_fields(
    const DeepSeekRankArtifactMetadataChunk& chunk) {
  const auto& fields = chunk.fields();
  return {fields.engine_epoch,
          fields.worker_generation,
          fields.world_size,
          fields.rank,
          fields.chunk_index,
          fields.chunk_count,
          fields.payload_offset + fields.payload_bytes,
          fields.total_blob_bytes,
          fields.process_manifest_identity,
          fields.process_identity,
          fields.pidfd_identity,
          fields.control_identity,
          fields.challenge_identity,
          fields.transfer_manifest_root,
          fields.descriptor_transfer_transaction_root,
          fields.metadata_root,
          fields.metadata_transaction_root,
          fields.blob_sha256,
          fields.chunk_sha256};
}

TEST(DeepSeekRankArtifactMetadataChunkTest,
     RoundTripsChunkAndRejectsEveryByteMutation) {
  auto payload = chunk_payload(37);
  auto chunk = DeepSeekRankArtifactMetadataChunk::Create(
      chunk_fields(payload, payload.size(), 0, 1), payload);
  ASSERT_TRUE(chunk.ok()) << chunk.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactMetadataChunkAbi,
            "pih_deepseek_rank_artifact_metadata_chunk_v1");
  EXPECT_EQ(kDeepSeekRankArtifactMetadataChunkFrameAbi,
            "pih_deepseek_rank_artifact_metadata_chunk_frame_v1");
  EXPECT_EQ(kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes,
            60U * 1024U);
  EXPECT_LT(kDeepSeekRankArtifactMetadataChunkFrameMaximumBytes,
            64U * 1024U);

  auto frame = encode_deepseek_rank_artifact_metadata_chunk(*chunk);
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  auto decoded = decode_deepseek_rank_artifact_metadata_chunk(*frame);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->fields().payload_offset, 0U);
  EXPECT_EQ(decoded->payload().size(), payload.size());
  EXPECT_EQ(decoded->frame_root(), chunk->frame_root());

  for (std::size_t index = 0; index < frame->size(); ++index) {
    auto corrupted = *frame;
    corrupted[index] ^= std::byte{1};
    EXPECT_FALSE(
        decode_deepseek_rank_artifact_metadata_chunk(corrupted).ok())
        << "accepted corrupted chunk byte " << index;
  }
}

TEST(DeepSeekRankArtifactMetadataChunkTest,
     RoundTripsExactAckAndRejectsEveryByteMutation) {
  auto payload = chunk_payload(37);
  auto chunk = DeepSeekRankArtifactMetadataChunk::Create(
                   chunk_fields(payload, payload.size(), 0, 1), payload)
                   .value();
  auto ack = DeepSeekRankArtifactMetadataChunkAck::Create(
      ack_fields(chunk));
  ASSERT_TRUE(ack.ok()) << ack.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactMetadataChunkAckAbi,
            "pih_deepseek_rank_artifact_metadata_chunk_ack_v1");
  EXPECT_EQ(kDeepSeekRankArtifactMetadataChunkAckFrameAbi,
            "pih_deepseek_rank_artifact_metadata_chunk_ack_frame_v1");

  auto frame = encode_deepseek_rank_artifact_metadata_chunk_ack(*ack);
  EXPECT_EQ(frame.size(),
            kDeepSeekRankArtifactMetadataChunkAckFrameBytes);
  auto decoded = decode_deepseek_rank_artifact_metadata_chunk_ack(frame);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->fields().cumulative_payload_bytes, payload.size());
  EXPECT_EQ(decoded->ack_root(), ack->ack_root());

  for (std::size_t index = 0; index < frame.size(); ++index) {
    auto corrupted = frame;
    corrupted[index] ^= std::byte{1};
    EXPECT_FALSE(
        decode_deepseek_rank_artifact_metadata_chunk_ack(corrupted).ok())
        << "accepted corrupted ACK byte " << index;
  }
}

TEST(DeepSeekRankArtifactMetadataChunkTest,
     EnforcesMaximumFrameAndCanonicalMultiChunkGeometry) {
  constexpr std::uint64_t kTotal =
      kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes + 1ULL;
  auto first_payload = chunk_payload(
      kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes);
  auto first_fields = chunk_fields(first_payload, kTotal, 0, 2);
  auto first = DeepSeekRankArtifactMetadataChunk::Create(
      first_fields, first_payload);
  ASSERT_TRUE(first.ok()) << first.status().message();
  auto first_frame =
      encode_deepseek_rank_artifact_metadata_chunk(*first).value();
  EXPECT_EQ(first_frame.size(),
            kDeepSeekRankArtifactMetadataChunkFrameMaximumBytes);

  auto final_payload = chunk_payload(1, 23);
  auto final_fields = chunk_fields(final_payload, kTotal, 1, 2);
  auto final = DeepSeekRankArtifactMetadataChunk::Create(
      final_fields, final_payload);
  ASSERT_TRUE(final.ok()) << final.status().message();
  EXPECT_TRUE(decode_deepseek_rank_artifact_metadata_chunk(
                  encode_deepseek_rank_artifact_metadata_chunk(*final)
                      .value())
                  .ok());

  auto short_nonfinal = first_fields;
  auto short_payload = chunk_payload(
      kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes - 1U);
  short_nonfinal.payload_bytes =
      static_cast<std::uint32_t>(short_payload.size());
  short_nonfinal.chunk_sha256 = sha256(short_payload).value();
  EXPECT_FALSE(DeepSeekRankArtifactMetadataChunk::Create(
                   short_nonfinal, std::move(short_payload))
                   .ok());

  auto wrong_offset = final_fields;
  --wrong_offset.payload_offset;
  EXPECT_FALSE(DeepSeekRankArtifactMetadataChunk::Create(
                   wrong_offset, final_payload)
                   .ok());
  auto wrong_count = final_fields;
  ++wrong_count.chunk_count;
  EXPECT_FALSE(DeepSeekRankArtifactMetadataChunk::Create(
                   wrong_count, final_payload)
                   .ok());
  auto wrong_hash = final_fields;
  wrong_hash.chunk_sha256 = chunk_digest(99);
  EXPECT_FALSE(DeepSeekRankArtifactMetadataChunk::Create(
                   wrong_hash, final_payload)
                   .ok());

  auto wrong_ack = ack_fields(*final);
  --wrong_ack.cumulative_payload_bytes;
  EXPECT_FALSE(
      DeepSeekRankArtifactMetadataChunkAck::Create(wrong_ack).ok());

  auto truncated = first_frame;
  truncated.pop_back();
  EXPECT_FALSE(
      decode_deepseek_rank_artifact_metadata_chunk(truncated).ok());
  auto trailing = first_frame;
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(decode_deepseek_rank_artifact_metadata_chunk(trailing).ok());
  auto oversized = std::vector<std::byte>(
      kDeepSeekRankArtifactMetadataChunkFrameMaximumBytes + 1U);
  EXPECT_FALSE(decode_deepseek_rank_artifact_metadata_chunk(oversized).ok());
}

}  // namespace
}  // namespace pih
