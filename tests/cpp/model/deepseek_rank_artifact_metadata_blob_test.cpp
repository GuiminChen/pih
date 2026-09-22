#include "pih/model/deepseek_rank_artifact_metadata_blob.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <gtest/gtest.h>

#include "deepseek_target_artifact_test_fixture.h"
#include "pih/model/deepseek_controller_artifact_catalog.h"

namespace pih {
namespace {

Sha256Digest metadata_digest(std::uint8_t seed) {
  Sha256Digest result{};
  for (std::size_t index = 0; index < result.bytes.size(); ++index) {
    result.bytes[index] = static_cast<std::byte>(seed + index);
  }
  return result;
}

struct MetadataFixture final {
  DeepSeekRankArtifactMetadataBlobFields fields;
  DeepSeekRankMappingPlan mapping;
  std::vector<DeepSeekRankTensorRecord> tensors;
};

MetadataFixture metadata_fixture(
    const DeepSeekRankArtifactHandoffPlan& handoff) {
  const auto& manifest = handoff.rank_manifest(0);
  const auto mapping_root =
      compile_deepseek_rank_mapping_plan_root(manifest.mapping()).value();
  const auto descriptor_root =
      compile_deepseek_rank_artifact_descriptor_handoff_root(
          manifest.descriptor_expectations())
          .value();
  const auto tensor_root = compile_deepseek_rank_tensor_handoff_root(
                               manifest.tensor_records())
                               .value();
  return {{7,
           8,
           1,
           0,
           false,
           10,
           100,
           200,
           300,
           400,
           static_cast<std::uint32_t>(
               manifest.descriptor_expectations().size()),
           metadata_digest(1),
           metadata_digest(2),
           metadata_digest(3),
           manifest.manifest_root(),
           handoff.artifact_root(),
           handoff.mapping_root(),
           mapping_root,
           descriptor_root,
           tensor_root},
          manifest.mapping(),
          std::vector<DeepSeekRankTensorRecord>(
              manifest.tensor_records().begin(),
              manifest.tensor_records().end())};
}

TEST(DeepSeekRankArtifactMetadataBlobTest,
     RoundTripsCanonicalRankMappingAndEveryTensorByte) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-rank-artifact-metadata-blob-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  auto catalog = DeepSeekControllerArtifactCatalog::OpenTargetGeneration(
                     root, test_fixture::reduced_target_artifact_root(),
                     pipeline, 4096,
                     ArtifactImmutabilityMode::kUncalibrated)
                     .value();
  auto handoff = DeepSeekRankArtifactHandoffPlan::Compile(catalog).value();
  auto fixture = metadata_fixture(handoff);
  auto blob = DeepSeekRankArtifactMetadataBlob::Create(
      fixture.fields, fixture.mapping, fixture.tensors);
  ASSERT_TRUE(blob.ok()) << blob.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactMetadataBlobAbi,
            "pih_deepseek_rank_artifact_metadata_blob_v1");
  EXPECT_EQ(kDeepSeekRankArtifactMetadataBlobFrameAbi,
            "pih_deepseek_rank_artifact_metadata_blob_frame_v1");
  EXPECT_EQ(kDeepSeekRankArtifactMetadataBlobMaximumBytes,
            128ULL * 1024ULL * 1024ULL);
  auto frame = encode_deepseek_rank_artifact_metadata_blob(*blob);
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  auto decoded = decode_deepseek_rank_artifact_metadata_blob(*frame);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->metadata_root(), blob->metadata_root());
  EXPECT_EQ(decoded->fields().artifact_handoff_rank_root,
            handoff.rank_root(0));
  EXPECT_EQ(decoded->mapping().logical_tensor_bytes,
            fixture.mapping.logical_tensor_bytes);
  ASSERT_EQ(decoded->tensor_records().size(), 1U);
  EXPECT_EQ(decoded->tensor_records()[0].tensor_name, "embed.weight");

  for (std::size_t index = 0; index < frame->size(); ++index) {
    auto corrupted = *frame;
    corrupted[index] ^= std::byte{1};
    EXPECT_FALSE(decode_deepseek_rank_artifact_metadata_blob(corrupted).ok())
        << "accepted corrupted byte " << index;
  }
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactMetadataBlobTest,
     RejectsTensorOutsideMappingAndNonExactFrames) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-rank-artifact-metadata-blob-negative-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  auto catalog = DeepSeekControllerArtifactCatalog::OpenTargetGeneration(
                     root, test_fixture::reduced_target_artifact_root(),
                     pipeline, 4096,
                     ArtifactImmutabilityMode::kUncalibrated)
                     .value();
  auto handoff = DeepSeekRankArtifactHandoffPlan::Compile(catalog).value();
  auto fixture = metadata_fixture(handoff);
  auto valid = DeepSeekRankArtifactMetadataBlob::Create(
                   fixture.fields, fixture.mapping, fixture.tensors)
                   .value();
  auto frame = encode_deepseek_rank_artifact_metadata_blob(valid).value();
  auto truncated = frame;
  truncated.pop_back();
  EXPECT_FALSE(decode_deepseek_rank_artifact_metadata_blob(truncated).ok());
  auto trailing = frame;
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(decode_deepseek_rank_artifact_metadata_blob(trailing).ok());

  fixture.tensors[0].file_begin = fixture.mapping.intervals[0].file_end;
  fixture.tensors[0].file_end = fixture.tensors[0].file_begin + 4;
  EXPECT_FALSE(DeepSeekRankArtifactMetadataBlob::Create(
                   fixture.fields, fixture.mapping,
                   std::move(fixture.tensors))
                   .ok());
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace pih
