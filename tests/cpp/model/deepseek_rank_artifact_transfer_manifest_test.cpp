#include "pih/model/deepseek_rank_artifact_transfer_manifest.h"

#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest transfer_digest(std::uint8_t seed) {
  Sha256Digest value{};
  value.bytes.fill(static_cast<std::byte>(seed));
  return value;
}

DeepSeekRankArtifactTransferManifestFields transfer_fields() {
  return {7,
          8,
          4,
          2,
          20,
          100,
          200,
          300,
          400,
          900,
          ArtifactImmutabilityMode::kUncalibrated,
          false,
          33,
          72'317,
          kDeepSeekRankArtifactTransferDescriptorBatchMaximum,
          3,
          transfer_digest(20),
          transfer_digest(21),
          transfer_digest(22),
          transfer_digest(23),
          transfer_digest(24),
          transfer_digest(25),
          transfer_digest(26),
          transfer_digest(27)};
}

TEST(DeepSeekRankArtifactTransferManifestTest,
     RoundTripsExactFixedFrameAndCanonicalRoot) {
  auto manifest =
      DeepSeekRankArtifactTransferManifest::Create(transfer_fields());
  ASSERT_TRUE(manifest.ok()) << manifest.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactTransferManifestAbi,
            "pih_deepseek_rank_artifact_transfer_manifest_v1");
  EXPECT_EQ(kDeepSeekRankArtifactTransferManifestFrameAbi,
            "pih_deepseek_rank_artifact_transfer_manifest_frame_v1");
  EXPECT_NE(manifest->manifest_root(), Sha256Digest{});
  EXPECT_EQ(manifest->manifest_root().hex(),
            "9b7b0b40d0f21f4862af4b4b47d4db5fb9ea93b1ddea7e805594abbb9b66aed5");
  const auto frame =
      encode_deepseek_rank_artifact_transfer_manifest(*manifest);
  EXPECT_EQ(frame.size(), kDeepSeekRankArtifactTransferManifestBytes);
  auto decoded = decode_deepseek_rank_artifact_transfer_manifest(frame);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->manifest_root(), manifest->manifest_root());
  const auto& fields = decoded->fields();
  EXPECT_EQ(fields.engine_epoch, 7U);
  EXPECT_EQ(fields.worker_generation, 8U);
  EXPECT_EQ(fields.world_size, 4U);
  EXPECT_EQ(fields.rank, 2U);
  EXPECT_EQ(fields.descriptor_count, 33U);
  EXPECT_EQ(fields.tensor_record_count, 72'317U);
  EXPECT_EQ(fields.descriptor_batch_maximum, 16U);
  EXPECT_EQ(fields.descriptor_batch_count, 3U);
  EXPECT_EQ(fields.artifact_admission_binding_root, transfer_digest(23));
  EXPECT_EQ(fields.artifact_root, transfer_digest(26));
}

TEST(DeepSeekRankArtifactTransferManifestTest,
     RejectsFrameShapeHeaderBooleanAndRootDrift) {
  auto manifest =
      DeepSeekRankArtifactTransferManifest::Create(transfer_fields()).value();
  auto frame = encode_deepseek_rank_artifact_transfer_manifest(manifest);
  EXPECT_FALSE(decode_deepseek_rank_artifact_transfer_manifest(
                   std::span(frame).first(frame.size() - 1U))
                   .ok());
  std::vector<std::byte> trailing(frame.begin(), frame.end());
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(decode_deepseek_rank_artifact_transfer_manifest(trailing).ok());

  for (const auto offset : std::array<std::size_t, 3>{0, 4, 6}) {
    auto changed = frame;
    changed[offset] ^= std::byte{1};
    EXPECT_FALSE(decode_deepseek_rank_artifact_transfer_manifest(changed).ok());
  }
  auto noncanonical_boolean = frame;
  noncanonical_boolean[84] = std::byte{2};
  EXPECT_FALSE(
      decode_deepseek_rank_artifact_transfer_manifest(noncanonical_boolean)
          .ok());
  auto changed_root = frame;
  changed_root.back() ^= std::byte{1};
  EXPECT_FALSE(
      decode_deepseek_rank_artifact_transfer_manifest(changed_root).ok());
}

TEST(DeepSeekRankArtifactTransferManifestTest,
     RejectsInvalidGeometryAndImmutabilityClaims) {
  auto fields = transfer_fields();
  fields.rank = fields.world_size;
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields = transfer_fields();
  fields.descriptor_count = 0;
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields = transfer_fields();
  fields.descriptor_count = 51;
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields = transfer_fields();
  fields.descriptor_batch_count = 2;
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields = transfer_fields();
  fields.tensor_record_count = 72'318;
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields = transfer_fields();
  fields.model_startup_rank_seed_root = {};
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields = transfer_fields();
  fields.artifact_admission_binding_root = {};
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields = transfer_fields();
  fields.source_catalog_production_eligible = true;
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields.immutability_mode = ArtifactImmutabilityMode::kFsVerity;
  EXPECT_TRUE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields.immutability_mode = ArtifactImmutabilityMode::kDmVeritySnapshot;
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
  fields = transfer_fields();
  fields.immutability_mode = static_cast<ArtifactImmutabilityMode>(255);
  EXPECT_FALSE(DeepSeekRankArtifactTransferManifest::Create(fields).ok());
}

TEST(DeepSeekRankArtifactTransferManifestTest,
     ProducesStableDistinctRootsForEverySupportedRank) {
  std::set<std::string> roots;
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      auto fields = transfer_fields();
      fields.world_size = world_size;
      fields.rank = rank;
      fields.process_manifest_identity += rank;
      fields.process_identity += rank;
      fields.pidfd_identity += rank;
      fields.control_identity += rank;
      fields.challenge_identity += rank;
      fields.model_startup_rank_seed_root =
          transfer_digest(static_cast<std::uint8_t>(30U + world_size * 4U +
                                                    rank));
      auto first = DeepSeekRankArtifactTransferManifest::Create(fields);
      auto second = DeepSeekRankArtifactTransferManifest::Create(fields);
      ASSERT_TRUE(first.ok()) << first.status().message();
      ASSERT_TRUE(second.ok()) << second.status().message();
      EXPECT_EQ(first->manifest_root(), second->manifest_root());
      roots.insert(first->manifest_root().hex());
    }
  }
  EXPECT_EQ(roots.size(), 10U);
}

}  // namespace
}  // namespace pih
