#include "pih/model/deepseek_rank_post_mapping_resource_codec.h"
#include "pih/model/deepseek_rank_post_mapping_resource_report.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest post_mapping_digest(std::uint8_t seed) {
  Sha256Digest value{};
  for (std::size_t index = 0; index < value.bytes.size(); ++index) {
    value.bytes[index] = static_cast<std::byte>(seed + index);
  }
  return value;
}

DeepSeekRankPostMappingResourceObservation post_mapping_observation() {
  return {{7,
           8,
           0,
           100,
           400,
           post_mapping_digest(1),
           post_mapping_digest(2),
           8,
           20,
           0,
           200,
           24,
           100,
           1000,
           210,
           true},
          post_mapping_digest(3),
          post_mapping_digest(4),
          post_mapping_digest(5),
          post_mapping_digest(6),
          post_mapping_digest(7),
          4096,
          ArtifactImmutabilityMode::kFsVerity,
          true,
          false};
}

DeepSeekRankPostMappingResourceAuthority post_mapping_authority() {
  DeepSeekRankPostMappingResourceAuthority value{};
  value.protocol_version = 1;
  value.engine_epoch = 7;
  value.worker_generation = 8;
  value.world_size = 1;
  value.rank = 0;
  value.process_manifest_identity = 20;
  value.process_identity = 100;
  value.pidfd_identity = 200;
  value.control_identity = 300;
  value.challenge_identity = 400;
  value.manifest_root = post_mapping_digest(1);
  value.spawn_resource_plan_root = post_mapping_digest(2);
  value.capacity_plan_instance_root = post_mapping_digest(3);
  value.os_resource_envelope_root = post_mapping_digest(4);
  value.first_resource_seal_root = post_mapping_digest(5);
  value.first_resource_plan_set_root = post_mapping_digest(6);
  value.descriptor_transaction_root = post_mapping_digest(7);
  value.metadata_transaction_root = post_mapping_digest(8);
  value.metadata_root = post_mapping_digest(9);
  value.expected_mapping_owner_root = post_mapping_digest(10);
  value.deadline_ns = 900;
  value.expected_mapped_interval_bytes = 4096;
  value.expected_immutability_mode = ArtifactImmutabilityMode::kFsVerity;
  value.expected_source_catalog_production_eligible = true;
  value.dspark_enabled = false;
  value.resource_plan = {8, 20, 200, 2, 4, 10,
                         value.os_resource_envelope_root};
  return value;
}

TEST(DeepSeekRankPostMappingResourceCodecTest,
     RoundTripsRootedObservationAndRejectsEveryMutation) {
  EXPECT_EQ(
      kDeepSeekRankPostMappingResourceObservationFrameAbi,
      "pih_deepseek_rank_post_mapping_resource_observation_frame_v1");
  auto encoded = encode_deepseek_rank_post_mapping_resource_observation(
      post_mapping_observation());
  ASSERT_TRUE(encoded.ok()) << encoded.status().message();
  static_assert(kDeepSeekRankPostMappingResourceObservationFrameBytes ==
                388);
  auto decoded = decode_deepseek_rank_post_mapping_resource_observation(
      *encoded);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->resources.engine_epoch, 7U);
  EXPECT_EQ(decoded->resources.open_fd_count, 20U);
  EXPECT_EQ(decoded->resources.scm_rights_inflight_fd_count, 0U);
  EXPECT_EQ(decoded->mapping_owner_root, post_mapping_digest(7));
  EXPECT_EQ(decoded->mapped_interval_bytes, 4096U);
  EXPECT_EQ(decoded->immutability_mode,
            ArtifactImmutabilityMode::kFsVerity);
  EXPECT_TRUE(decoded->source_catalog_production_eligible);
  EXPECT_FALSE(decoded->dspark_enabled);

  for (std::size_t index = 0; index < encoded->size(); ++index) {
    auto changed = *encoded;
    changed[index] ^= std::byte{1};
    EXPECT_FALSE(
        decode_deepseek_rank_post_mapping_resource_observation(changed)
            .ok())
        << "accepted observation mutation at byte " << index;
  }
  EXPECT_FALSE(decode_deepseek_rank_post_mapping_resource_observation(
                   std::span<const std::byte>(*encoded).first(
                       encoded->size() - 1))
                   .ok());
  std::vector<std::byte> trailing(encoded->begin(), encoded->end());
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(
      decode_deepseek_rank_post_mapping_resource_observation(trailing).ok());
}

TEST(DeepSeekRankPostMappingResourceCodecTest,
     RoundTripsRootedAuthorityAndRejectsEveryMutation) {
  EXPECT_EQ(
      kDeepSeekRankPostMappingResourceAuthorityFrameAbi,
      "pih_deepseek_rank_post_mapping_resource_authority_frame_v1");
  auto encoded = encode_deepseek_rank_post_mapping_resource_authority(
      post_mapping_authority());
  ASSERT_TRUE(encoded.ok()) << encoded.status().message();
  static_assert(kDeepSeekRankPostMappingResourceAuthorityFrameBytes == 500);
  auto decoded = decode_deepseek_rank_post_mapping_resource_authority(
      *encoded);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->protocol_version, 1U);
  EXPECT_EQ(decoded->world_size, 1U);
  EXPECT_EQ(decoded->process_identity, 100U);
  EXPECT_EQ(decoded->first_resource_plan_set_root,
            post_mapping_digest(6));
  EXPECT_EQ(decoded->expected_mapping_owner_root,
            post_mapping_digest(10));
  EXPECT_EQ(decoded->resource_plan.worker_vma_peak, 200U);
  EXPECT_EQ(decoded->resource_plan.os_resource_envelope_root,
            decoded->os_resource_envelope_root);

  for (std::size_t index = 0; index < encoded->size(); ++index) {
    auto changed = *encoded;
    changed[index] ^= std::byte{1};
    EXPECT_FALSE(
        decode_deepseek_rank_post_mapping_resource_authority(changed).ok())
        << "accepted authority mutation at byte " << index;
  }
  EXPECT_FALSE(decode_deepseek_rank_post_mapping_resource_authority(
                   std::span<const std::byte>(*encoded).first(
                       encoded->size() - 1))
                   .ok());
  auto invalid = post_mapping_authority();
  invalid.expected_immutability_mode =
      static_cast<ArtifactImmutabilityMode>(3);
  EXPECT_FALSE(
      encode_deepseek_rank_post_mapping_resource_authority(invalid).ok());
}

TEST(DeepSeekRankPostMappingResourceSealTest, PublishesFrozenAbiNames) {
  EXPECT_EQ(kDeepSeekRankPostMappingResourceReceiptAbi,
            "pih_deepseek_rank_post_mapping_resource_receipt_v1");
  EXPECT_EQ(kDeepSeekRankPostMappingResourceSealAbi,
            "pih_deepseek_rank_post_mapping_resource_seal_v1");
}

TEST(DeepSeekRankPostMappingResourceReportTest,
     CommitsExactAuthorityAndObservationBeforeAllRankAdmission) {
  const auto authority = post_mapping_authority();
  auto observation = post_mapping_observation();
  observation.resources.acknowledged_capacity_plan_instance_root =
      authority.capacity_plan_instance_root;
  observation.resources.acknowledged_os_resource_envelope_root =
      authority.os_resource_envelope_root;
  observation.first_resource_seal_root =
      authority.first_resource_seal_root;
  observation.descriptor_transaction_root =
      authority.descriptor_transaction_root;
  observation.metadata_transaction_root =
      authority.metadata_transaction_root;
  observation.metadata_root = authority.metadata_root;
  observation.mapping_owner_root =
      authority.expected_mapping_owner_root;
  observation.mapped_interval_bytes =
      authority.expected_mapped_interval_bytes;
  observation.immutability_mode = authority.expected_immutability_mode;
  observation.source_catalog_production_eligible =
      authority.expected_source_catalog_production_eligible;
  observation.dspark_enabled = authority.dspark_enabled;
  auto report = DeepSeekRankPostMappingResourceReport::Compile(
      authority, observation);
  ASSERT_TRUE(report.ok()) << report.status().message();
  EXPECT_EQ(kDeepSeekRankPostMappingResourceReportAbi,
            "pih_deepseek_rank_post_mapping_resource_report_v1");
  EXPECT_EQ(report->engine_epoch(), 7U);
  EXPECT_EQ(report->worker_generation(), 8U);
  EXPECT_EQ(report->world_size(), 1U);
  EXPECT_EQ(report->rank(), 0U);
  EXPECT_EQ(report->process_identity(), 100U);
  EXPECT_EQ(report->mapping_owner_root(),
            authority.expected_mapping_owner_root);
  EXPECT_NE(report->authority_frame_root(), Sha256Digest{});
  EXPECT_NE(report->observation_frame_root(), Sha256Digest{});
  EXPECT_NE(report->report_root(), Sha256Digest{});

  auto replay = DeepSeekRankPostMappingResourceReport::Compile(
      authority, observation);
  ASSERT_TRUE(replay.ok()) << replay.status().message();
  EXPECT_EQ(replay->report_root(), report->report_root());

  auto different_deadline = authority;
  ++different_deadline.deadline_ns;
  auto changed_report = DeepSeekRankPostMappingResourceReport::Compile(
      different_deadline, observation);
  ASSERT_TRUE(changed_report.ok()) << changed_report.status().message();
  EXPECT_NE(changed_report->report_root(), report->report_root());

  auto changed_observation = observation;
  changed_observation.mapping_owner_root = post_mapping_digest(99);
  EXPECT_FALSE(DeepSeekRankPostMappingResourceReport::Compile(
                   authority, changed_observation)
                   .ok());
  changed_observation = observation;
  changed_observation.resources.scm_rights_inflight_fd_count = 1;
  EXPECT_FALSE(DeepSeekRankPostMappingResourceReport::Compile(
                   authority, changed_observation)
                   .ok());
}

}  // namespace
}  // namespace pih
