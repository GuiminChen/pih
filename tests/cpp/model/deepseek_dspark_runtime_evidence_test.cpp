#include "pih/model/deepseek_dspark_runtime_evidence.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "pih/model/profile_authority_codec.h"
#include "pih/model/runtime_evidence_projection.h"
#include "pih/model/runtime_profile_authority_lease.h"
#include "pih/model/runtime_profile_device_gate.h"
#include "pih/model/runtime_profile_reference_closure.h"
#include "pih/model/runtime_profile_reference_lease.h"
#include "pih/model/deepseek_runtime_artifact_evidence.h"
#include "pih/model/sealed_reachable_object_dag.h"

namespace pih {
namespace {

Sha256Digest root(std::uint8_t seed) {
  Sha256Digest value{};
  value.bytes.fill(static_cast<std::byte>(seed));
  return value;
}

DeepSeekDsparkResolvedInputRoots resolved_inputs() {
  return {root(1), root(2), root(3), root(5), root(6), root(7)};
}

DeepSeekDsparkQualificationRoots qualification_roots() {
  return {root(21), root(22), root(23), root(24)};
}

RuntimeProfileRoots profile_roots(const Sha256Digest& release_root) {
  const auto inputs = resolved_inputs();
  return {inputs.runtime_semantic_root,
          inputs.capacity_template_root,
          inputs.feature_selection_root,
          release_root,
          inputs.hardware_identity_root,
          inputs.kernel_closure_root,
          inputs.security_runtime_root};
}

class AcceptingVerifier final : public ProfileEd25519Verifier {
 public:
  Status verify(std::string_view, std::span<const std::byte>,
                std::span<const std::byte>,
                std::span<const std::byte>) override {
    return Status::Ok();
  }
};

class H100Probe final : public RuntimeProfileDeviceProbe {
 public:
  Result<RuntimeProfileDeviceObservation> observe(
      std::int32_t ordinal) override {
    RuntimeProfileDeviceObservation result{};
    result.ordinal = ordinal;
    result.uuid[0] = static_cast<std::byte>(ordinal + 1);
    result.name = "NVIDIA H100 PCIe";
    result.compute_major = 9;
    result.compute_minor = 0;
    result.total_memory_bytes = 80'000'000'000ULL;
    return result;
  }
};

class ReferenceOwner final : public RuntimeProfileReferenceLeaseOwner {};
class ReferenceProbe final : public RuntimeProfileReferenceLeaseProbe {
 public:
  Result<RuntimeProfileReferenceLeaseObservation> observe(
      const RuntimeProfileReferenceDescriptor& descriptor) override {
    return RuntimeProfileReferenceLeaseObservation{
        descriptor.role, descriptor.exact_bytes, descriptor.object_root,
        true, true, std::make_shared<ReferenceOwner>()};
  }
};

class AuthorityOwner final : public RuntimeProfileAuthorityLeaseOwner {};
class AuthorityProbe final : public RuntimeProfileAuthorityLeaseProbe {
 public:
  Result<RuntimeProfileAuthorityLeaseObservation> observe(
      const RuntimeProfileAuthorityDescriptor& descriptor) override {
    return RuntimeProfileAuthorityLeaseObservation{
        descriptor.role, descriptor.exact_bytes, descriptor.content_root,
        true, true, std::make_shared<AuthorityOwner>()};
  }
};

ProfileSignature signature() {
  return {"release-1",
          std::vector<std::byte>(kEd25519SignatureBytes, std::byte{0x5a})};
}

RuntimeProfileReferenceClosure closure_for(const RuntimeProfileRoots& roots) {
  return RuntimeProfileReferenceClosure::Create({
      {RuntimeProfileReferenceRole::kRuntimeSemantic, "semantic_v1", 1,
       roots.runtime_semantic_root},
      {RuntimeProfileReferenceRole::kCapacityTemplate, "capacity_v1", 2,
       roots.capacity_template_root},
      {RuntimeProfileReferenceRole::kFeatureSelection, "feature_v1", 3,
       roots.feature_selection_root},
      {RuntimeProfileReferenceRole::kReleaseEvidence, "evidence_v1", 4,
       roots.release_evidence_root},
      {RuntimeProfileReferenceRole::kHardwareIdentity, "hardware_v1", 5,
       roots.hardware_identity_root},
      {RuntimeProfileReferenceRole::kKernelClosure, "kernel_v1", 6,
       roots.kernel_closure_root},
      {RuntimeProfileReferenceRole::kSecurityRuntime, "security_v1", 7,
       roots.security_runtime_root},
  }).value();
}

RuntimeEngineAdmission production_admission(
    std::uint8_t world_size, const RuntimeEvidenceProjection& projection) {
  const auto roots = profile_roots(projection.projection_root());
  auto closure = closure_for(roots);
  auto payload = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative,
      RuntimeProfileGpuFamily::kH100Pcie80GiB, world_size, true,
      RuntimeProfileResidency::kHostSpill,
      RuntimeProfileEvidenceState::kCorrectnessSupported, roots).value();
  auto envelope = SignedProfileEnvelope::Create(
      "deepseek-dspark-h100", "r1", 9, payload.canonical_bytes(),
      closure.closure_root(), {signature()}).value();
  auto graph = verify_sealed_reachable_object_dag(
      {{"schema_v1", root(40), 1, 1, true, {}}}, {root(40)}).value();
  auto catalog = SignedProfileCatalog::CreateGraphBound(
      9, {graph.snapshot_root(), graph.node_count(), graph.edge_count()},
      {{"deepseek-dspark-h100", "r1", envelope.envelope_root(),
        envelope.envelope_bytes()}},
      {signature()}).value();
  ProfileTrustKey key{};
  key.key_id = "release-1";
  key.role = ProfileSignerRole::kRelease;
  key.public_key.fill(std::byte{0x22});
  auto policy = ProfileTrustPolicy::Create(9, catalog.catalog_root(), {key})
                    .value();
  AcceptingVerifier verifier;
  auto authority = verify_profile_authority(
      policy, catalog, envelope, "deepseek-dspark-h100", "r1", verifier)
                       .value();
  auto profile = bind_verified_runtime_profile(authority, envelope).value();
  std::vector<std::int32_t> ordinals(world_size);
  for (std::uint8_t index = 0; index < world_size; ++index) {
    ordinals[index] = index;
  }
  H100Probe device_probe;
  auto devices = verify_runtime_profile_devices(profile, ordinals,
                                                 device_probe).value();
  auto admission = admit_runtime_engine(
      profile, devices, RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative).value();
  admission = bind_runtime_engine_evidence_projection(
      std::move(admission), projection).value();
  ReferenceProbe reference_probe;
  auto leases = verify_runtime_profile_reference_leases(
      closure, reference_probe).value();
  admission = retain_runtime_engine_reference_leases(
      std::move(admission), std::move(leases)).value();
  AuthorityProbe authority_probe;
  auto authority_leases = verify_runtime_profile_authority_leases(
      policy, catalog, envelope, closure, graph, authority_probe).value();
  return retain_runtime_engine_authority_leases(
      std::move(admission), std::move(authority_leases)).value();
}

RuntimeEvidenceProjection projection_for(
    const DeepSeekDsparkRuntimeEvidenceManifest& manifest,
    std::uint8_t bundle_seed = 30) {
  return RuntimeEvidenceProjection::Create(
      RuntimeEvidenceBundleState::kComplete,
      RuntimeEvidenceReplayState::kPassed,
      RuntimeHardwareQualificationState::kSupported,
      {root(bundle_seed), root(31), root(32),
       manifest.domain_decision_roots_digest(), root(33)}).value();
}

TEST(DeepSeekDsparkRuntimeEvidenceTest,
     RoundTripsTheBoundedH100DecisionManifest) {
  auto manifest = DeepSeekDsparkRuntimeEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 4,
      RuntimeProfileResidency::kHostSpill, resolved_inputs(), root(20),
      qualification_roots());
  ASSERT_TRUE(manifest.ok()) << manifest.status().message();
  EXPECT_EQ(manifest->canonical_bytes().size(),
            DeepSeekDsparkRuntimeEvidenceManifest::kCanonicalBytes);
  auto parsed = parse_deepseek_dspark_runtime_evidence_manifest(
      manifest->canonical_bytes());
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(parsed->domain_decision_roots_digest(),
            manifest->domain_decision_roots_digest());
  EXPECT_EQ(parsed->world_size(), 4);

  auto trailing = std::vector<std::byte>(manifest->canonical_bytes().begin(),
                                         manifest->canonical_bytes().end());
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(parse_deepseek_dspark_runtime_evidence_manifest(trailing).ok());
  EXPECT_FALSE(DeepSeekDsparkRuntimeEvidenceManifest::Create(
                   RuntimeProfileGpuFamily::kRtx4090D24GiB, 1,
                   RuntimeProfileResidency::kHostSpill, resolved_inputs(),
                   root(20), qualification_roots()).ok());
  auto duplicate = qualification_roots();
  duplicate.exact_profile_hardware_receipt_root =
      duplicate.three_stage_runtime_numerical_receipt_root;
  EXPECT_FALSE(DeepSeekDsparkRuntimeEvidenceManifest::Create(
                   RuntimeProfileGpuFamily::kH100Pcie80GiB, 1,
                   RuntimeProfileResidency::kHostSpill, resolved_inputs(),
                   root(20), duplicate).ok());
}

TEST(DeepSeekDsparkRuntimeEvidenceTest,
     PermitJoinsProductionAdmissionArtifactAndExactWorldSize) {
  auto manifest = DeepSeekDsparkRuntimeEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 4,
      RuntimeProfileResidency::kHostSpill, resolved_inputs(), root(20),
      qualification_roots()).value();
  auto projection = projection_for(manifest);
  auto admission = production_admission(4, projection);
  auto permit = issue_deepseek_dspark_runtime_permit(
      admission, manifest, root(20));
  ASSERT_TRUE(permit.ok()) << permit.status().message();
  EXPECT_EQ(permit->world_size(), 4);
  EXPECT_TRUE(validate_deepseek_dspark_runtime_permit(
                  *permit, admission, root(20)).ok());
  EXPECT_FALSE(validate_deepseek_dspark_runtime_permit(
                   *permit, admission, root(25)).ok());

  auto pp3_manifest = DeepSeekDsparkRuntimeEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 3,
      RuntimeProfileResidency::kHostSpill, resolved_inputs(), root(20),
      qualification_roots()).value();
  auto pp3_admission = production_admission(3, projection_for(pp3_manifest));
  EXPECT_FALSE(validate_deepseek_dspark_runtime_permit(
                   *permit, pp3_admission, root(20)).ok());

  auto common_binding =
      DeepSeekRuntimeArtifactAdmissionBinding::Issue(admission, manifest);
  ASSERT_TRUE(common_binding.ok()) << common_binding.status().message();
  EXPECT_TRUE(common_binding->admission_authority_retained());
  EXPECT_EQ(common_binding->artifact_root(), root(20));
  EXPECT_TRUE(common_binding->validate(admission, root(20)).ok());
  EXPECT_FALSE(common_binding->validate(pp3_admission, root(20)).ok());
}

TEST(DeepSeekDsparkRuntimeEvidenceTest,
     RejectsEvidenceThatWasNotCommittedByTheReleaseProjection) {
  auto manifest = DeepSeekDsparkRuntimeEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 1,
      RuntimeProfileResidency::kHostSpill, resolved_inputs(), root(20),
      qualification_roots()).value();
  auto admission = production_admission(1, projection_for(manifest));

  auto changed_roots = qualification_roots();
  changed_roots.official_checkpoint_numerical_receipt_root = root(29);
  auto changed = DeepSeekDsparkRuntimeEvidenceManifest::Create(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 1,
      RuntimeProfileResidency::kHostSpill, resolved_inputs(), root(20),
      changed_roots).value();
  EXPECT_FALSE(issue_deepseek_dspark_runtime_permit(
                   admission, changed, root(20)).ok());
  EXPECT_FALSE(issue_deepseek_dspark_runtime_permit(
                   admission, manifest, root(19)).ok());
}

}  // namespace
}  // namespace pih
