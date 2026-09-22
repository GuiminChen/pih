#include "pih/model/deepseek_runtime_artifact_evidence.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "deepseek_rank_capacity_test_fixture.h"
#include "pih/model/runtime_evidence_projection.h"

namespace pih {
namespace {

Sha256Digest artifact_evidence_digest(std::uint8_t seed) {
  return test_fixture::rank_capacity_digest(seed);
}

DeepSeekRuntimeResolvedInputRoots artifact_resolved_inputs() {
  return {artifact_evidence_digest(1), artifact_evidence_digest(2),
          artifact_evidence_digest(3), artifact_evidence_digest(5),
          artifact_evidence_digest(6), artifact_evidence_digest(7)};
}

DeepSeekRuntimeArtifactQualificationRoots artifact_qualifications() {
  return {artifact_evidence_digest(20), artifact_evidence_digest(21),
          artifact_evidence_digest(22)};
}

RuntimeEvidenceProjection artifact_projection(
    const DeepSeekRuntimeArtifactEvidenceManifest& manifest,
    std::uint8_t bundle_seed = 30) {
  return RuntimeEvidenceProjection::Create(
      RuntimeEvidenceBundleState::kComplete,
      RuntimeEvidenceReplayState::kPassed,
      RuntimeHardwareQualificationState::kSupported,
      {artifact_evidence_digest(bundle_seed), artifact_evidence_digest(31),
       artifact_evidence_digest(32),
       manifest.domain_decision_roots_digest(), artifact_evidence_digest(34)})
      .value();
}

TEST(DeepSeekRuntimeArtifactEvidenceTest,
     RoundTripsCanonicalMainPathEvidenceAndFreezesGoldenRoot) {
  auto manifest = DeepSeekRuntimeArtifactEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 4,
      RuntimeProfileResidency::kFullResident, artifact_resolved_inputs(),
      artifact_evidence_digest(19), artifact_qualifications());
  ASSERT_TRUE(manifest.ok()) << manifest.status().message();
  EXPECT_EQ(kDeepSeekRuntimeArtifactEvidenceManifestAbi,
            "pih_deepseek_runtime_artifact_evidence_manifest_v1");
  EXPECT_EQ(kDeepSeekRuntimeArtifactAdmissionBindingAbi,
            "pih_deepseek_runtime_artifact_admission_binding_v1");
  EXPECT_EQ(manifest->canonical_bytes().size(),
            DeepSeekRuntimeArtifactEvidenceManifest::kCanonicalBytes);
  EXPECT_EQ(manifest->domain_decision_roots_digest().hex(),
            "8b14275b7396825eb7fdd02f4f21519a794b8402e4f33674c88007bce3d5ab1f");

  auto parsed = parse_deepseek_runtime_artifact_evidence_manifest(
      manifest->canonical_bytes());
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(parsed->domain_decision_roots_digest(),
            manifest->domain_decision_roots_digest());
  EXPECT_EQ(parsed->artifact_root(), artifact_evidence_digest(19));

  auto truncated = std::vector<std::byte>(manifest->canonical_bytes().begin(),
                                           manifest->canonical_bytes().end() - 1);
  EXPECT_FALSE(
      parse_deepseek_runtime_artifact_evidence_manifest(truncated).ok());
  auto trailing = std::vector<std::byte>(manifest->canonical_bytes().begin(),
                                         manifest->canonical_bytes().end());
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(parse_deepseek_runtime_artifact_evidence_manifest(trailing).ok());
  auto wrong_version = std::vector<std::byte>(manifest->canonical_bytes().begin(),
                                               manifest->canonical_bytes().end());
  wrong_version[8] = std::byte{2};
  EXPECT_FALSE(
      parse_deepseek_runtime_artifact_evidence_manifest(wrong_version).ok());
  auto wrong_family = std::vector<std::byte>(manifest->canonical_bytes().begin(),
                                              manifest->canonical_bytes().end());
  wrong_family[9] = std::byte{255};
  EXPECT_FALSE(
      parse_deepseek_runtime_artifact_evidence_manifest(wrong_family).ok());
}

TEST(DeepSeekRuntimeArtifactEvidenceTest,
     AcceptsOnlyClosedMainPathHardwareRowsAndDistinctQualificationRoles) {
  for (std::uint8_t world_size = 1; world_size <= 4; ++world_size) {
    EXPECT_TRUE(DeepSeekRuntimeArtifactEvidenceManifest::Create(
                    RuntimeProfileGpuFamily::kRtx4090D24GiB, world_size,
                    RuntimeProfileResidency::kHostSpill,
                    artifact_resolved_inputs(), artifact_evidence_digest(19),
                    artifact_qualifications())
                    .ok());
    EXPECT_TRUE(DeepSeekRuntimeArtifactEvidenceManifest::Create(
                    RuntimeProfileGpuFamily::kH100Pcie80GiB, world_size,
                    RuntimeProfileResidency::kHostSpill,
                    artifact_resolved_inputs(), artifact_evidence_digest(19),
                    artifact_qualifications())
                    .ok());
    const auto full_resident =
        DeepSeekRuntimeArtifactEvidenceManifest::Create(
            RuntimeProfileGpuFamily::kH100Pcie80GiB, world_size,
            RuntimeProfileResidency::kFullResident,
            artifact_resolved_inputs(), artifact_evidence_digest(19),
            artifact_qualifications());
    EXPECT_EQ(full_resident.ok(), world_size >= 3);
  }
  EXPECT_FALSE(DeepSeekRuntimeArtifactEvidenceManifest::Create(
                   RuntimeProfileGpuFamily::kRtx4090D24GiB, 1,
                   RuntimeProfileResidency::kFullResident,
                   artifact_resolved_inputs(), artifact_evidence_digest(19),
                   artifact_qualifications())
                   .ok());
  auto duplicate = artifact_qualifications();
  duplicate.exact_profile_hardware_receipt_root =
      duplicate.official_checkpoint_numerical_receipt_root;
  EXPECT_FALSE(DeepSeekRuntimeArtifactEvidenceManifest::Create(
                   RuntimeProfileGpuFamily::kH100Pcie80GiB, 4,
                   RuntimeProfileResidency::kFullResident,
                   artifact_resolved_inputs(), artifact_evidence_digest(19),
                   duplicate)
                   .ok());
  EXPECT_FALSE(DeepSeekRuntimeArtifactEvidenceManifest::Create(
                   static_cast<RuntimeProfileGpuFamily>(255), 1,
                   RuntimeProfileResidency::kHostSpill,
                   artifact_resolved_inputs(), artifact_evidence_digest(19),
                   artifact_qualifications())
                   .ok());
  auto zero_input = artifact_resolved_inputs();
  zero_input.runtime_semantic_root = {};
  EXPECT_FALSE(DeepSeekRuntimeArtifactEvidenceManifest::Create(
                   RuntimeProfileGpuFamily::kH100Pcie80GiB, 1,
                   RuntimeProfileResidency::kHostSpill, zero_input,
                   artifact_evidence_digest(19), artifact_qualifications())
                   .ok());
}

TEST(DeepSeekRuntimeArtifactEvidenceTest,
     BindingRetainsAndRevalidatesExactProductionAdmission) {
  auto manifest = DeepSeekRuntimeArtifactEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 4,
      RuntimeProfileResidency::kHostSpill, artifact_resolved_inputs(),
      artifact_evidence_digest(19), artifact_qualifications())
                      .value();
  auto projection = artifact_projection(manifest);
  auto admission = test_fixture::rank_capacity_admission(
      4, true, true, std::span<const std::int32_t>{}, &projection);
  auto binding =
      DeepSeekRuntimeArtifactAdmissionBinding::Issue(admission, manifest);
  ASSERT_TRUE(binding.ok()) << binding.status().message();
  EXPECT_TRUE(binding->admission_authority_retained());
  EXPECT_EQ(binding->artifact_root(), artifact_evidence_digest(19));
  EXPECT_EQ(binding->evidence_manifest_root(),
            manifest.domain_decision_roots_digest());
  EXPECT_NE(binding->binding_root(), Sha256Digest{});
  EXPECT_EQ(binding->binding_root().hex(),
            "e0e582072b74b03d8b0318570774b66dddbfc4d258066a1abfe49e2dd9913f50");
  EXPECT_TRUE(binding->validate(admission, artifact_evidence_digest(19)).ok());
  EXPECT_FALSE(binding->validate(admission, artifact_evidence_digest(18)).ok());

  auto foreign_projection = artifact_projection(manifest, 29);
  auto foreign_admission = test_fixture::rank_capacity_admission(
      4, true, true, std::span<const std::int32_t>{}, &foreign_projection);
  EXPECT_FALSE(
      binding->validate(foreign_admission, artifact_evidence_digest(19)).ok());

  auto unretained = test_fixture::rank_capacity_admission(
      4, false, true, std::span<const std::int32_t>{}, &projection);
  EXPECT_FALSE(
      DeepSeekRuntimeArtifactAdmissionBinding::Issue(unretained, manifest).ok());
}

TEST(DeepSeekRuntimeArtifactEvidenceTest,
     BindingRejectsForeignDomainAndResolvedInputsBeforeRetention) {
  auto manifest = DeepSeekRuntimeArtifactEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 3,
      RuntimeProfileResidency::kHostSpill, artifact_resolved_inputs(),
      artifact_evidence_digest(19), artifact_qualifications())
                      .value();
  auto projection = artifact_projection(manifest);
  auto admission = test_fixture::rank_capacity_admission(
      3, true, true, std::span<const std::int32_t>{}, &projection);

  auto changed_inputs = artifact_resolved_inputs();
  changed_inputs.feature_selection_root = artifact_evidence_digest(17);
  auto changed = DeepSeekRuntimeArtifactEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 3,
      RuntimeProfileResidency::kHostSpill, changed_inputs,
      artifact_evidence_digest(19), artifact_qualifications())
                     .value();
  EXPECT_FALSE(
      DeepSeekRuntimeArtifactAdmissionBinding::Issue(admission, changed).ok());

  auto wrong_world = DeepSeekRuntimeArtifactEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 2,
      RuntimeProfileResidency::kHostSpill, artifact_resolved_inputs(),
      artifact_evidence_digest(19), artifact_qualifications())
                         .value();
  EXPECT_FALSE(DeepSeekRuntimeArtifactAdmissionBinding::Issue(
                   admission, wrong_world)
                   .ok());
}

}  // namespace
}  // namespace pih
