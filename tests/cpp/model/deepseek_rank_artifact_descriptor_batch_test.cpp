#include "pih/model/deepseek_rank_artifact_descriptor_batch.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
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

std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations(
    std::uint32_t count, std::size_t name_bytes = 7) {
  std::vector<DeepSeekRankArtifactDescriptorExpectation> result;
  for (std::uint32_t ordinal = 0; ordinal < count; ++ordinal) {
    std::string name(name_bytes, static_cast<char>('a' + ordinal));
    result.push_back(
        {ordinal,
         std::move(name),
         {100U + ordinal, 200U + ordinal, 300U + ordinal,
          400 + ordinal, ordinal},
         ArtifactImmutabilityMode::kUncalibrated,
         {}});
  }
  return result;
}

DeepSeekRankArtifactDescriptorBatchFields fields_for(
    std::span<const DeepSeekRankArtifactDescriptorExpectation> descriptors) {
  const auto batch_root =
      compile_deepseek_rank_artifact_transfer_descriptor_batch_root(
          0, 0, descriptors)
          .value();
  const auto adopted_root =
      compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
          0, descriptors)
          .value();
  return {7,
          8,
          1,
          0,
          0,
          1,
          0,
          static_cast<std::uint32_t>(descriptors.size()),
          static_cast<std::uint32_t>(descriptors.size()),
          true,
          10,
          100,
          200,
          300,
          400,
          digest(1),
          digest(2),
          digest(3),
          batch_root,
          adopted_root};
}

TEST(DeepSeekRankArtifactDescriptorBatchTest,
     RoundTripsBoundedMetadataAndRejectsEveryByteMutation) {
  auto metadata = expectations(1);
  auto batch = DeepSeekRankArtifactDescriptorBatch::Create(
      fields_for(metadata), metadata);
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactDescriptorBatchAbi,
            "pih_deepseek_rank_artifact_descriptor_batch_v1");
  EXPECT_EQ(kDeepSeekRankArtifactDescriptorBatchFrameAbi,
            "pih_deepseek_rank_artifact_descriptor_batch_frame_v1");
  auto frame = encode_deepseek_rank_artifact_descriptor_batch(*batch);
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  EXPECT_EQ(frame->size(), 373U);
  auto decoded = decode_deepseek_rank_artifact_descriptor_batch(*frame);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  ASSERT_EQ(decoded->expectations().size(), 1U);
  EXPECT_EQ(decoded->expectations()[0].shard_name, metadata[0].shard_name);
  EXPECT_EQ(decoded->expectations()[0].identity, metadata[0].identity);
  EXPECT_EQ(decoded->frame_root(), batch->frame_root());

  for (std::size_t offset = 0; offset < frame->size(); ++offset) {
    auto changed = *frame;
    changed[offset] ^= std::byte{0x01};
    EXPECT_FALSE(
        decode_deepseek_rank_artifact_descriptor_batch(changed).ok())
        << "offset=" << offset;
  }
  auto trailing = *frame;
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(decode_deepseek_rank_artifact_descriptor_batch(trailing).ok());
  EXPECT_FALSE(decode_deepseek_rank_artifact_descriptor_batch(
                   std::span<const std::byte>(*frame).first(frame->size() - 1))
                   .ok());
}

TEST(DeepSeekRankArtifactDescriptorBatchTest,
     ReachesExactMaximumWithSixteenMaximumNames) {
  auto metadata = expectations(16, 255);
  auto batch = DeepSeekRankArtifactDescriptorBatch::Create(
      fields_for(metadata), metadata);
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  auto frame = encode_deepseek_rank_artifact_descriptor_batch(*batch);
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  EXPECT_EQ(frame->size(),
            kDeepSeekRankArtifactDescriptorBatchFrameMaximumBytes);
  auto decoded = decode_deepseek_rank_artifact_descriptor_batch(*frame);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->expectations().size(), 16U);
}

TEST(DeepSeekRankArtifactDescriptorBatchTest,
     RejectsMetadataRootCountOrderAndInitialAdoptionDrift) {
  auto metadata = expectations(2);
  auto fields = fields_for(metadata);
  fields.descriptor_count = 1;
  fields.cumulative_descriptor_count = 1;
  EXPECT_FALSE(
      DeepSeekRankArtifactDescriptorBatch::Create(fields, metadata).ok());
  fields = fields_for(metadata);
  fields.descriptor_batch_root = digest(91);
  EXPECT_FALSE(
      DeepSeekRankArtifactDescriptorBatch::Create(fields, metadata).ok());
  fields = fields_for(metadata);
  fields.adopted_descriptor_set_root = digest(92);
  EXPECT_FALSE(
      DeepSeekRankArtifactDescriptorBatch::Create(fields, metadata).ok());
  fields = fields_for(metadata);
  std::swap(metadata[0], metadata[1]);
  EXPECT_FALSE(
      DeepSeekRankArtifactDescriptorBatch::Create(fields, metadata).ok());
}

}  // namespace
}  // namespace pih
