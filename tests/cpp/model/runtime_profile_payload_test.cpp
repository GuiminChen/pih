#include "pih/model/runtime_profile_payload.h"
#include "pih/model/runtime_profile_authority_lease.h"
#include "pih/model/runtime_evidence_projection.h"
#include "pih/model/runtime_profile_readiness_receipt.h"
#include "pih/model/catalog_placement_transaction.h"
#include "pih/model/catalog_placement_recovery_codec.h"
#include "pih/model/catalog_placement_recovery_journal.h"
#include "pih/model/runtime_profile_device_gate.h"
#include "pih/model/runtime_profile_reference_lease.h"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <vector>

namespace pih {
namespace {

Sha256Digest runtime_root(std::uint8_t seed) {
  Sha256Digest value{};
  value.bytes.fill(static_cast<std::byte>(seed));
  return value;
}

RuntimeEvidenceProjection runtime_evidence_projection() {
  const RuntimeEvidenceRoots evidence_roots{
      runtime_root(30), runtime_root(31), runtime_root(32), runtime_root(33),
      runtime_root(34)};
  return RuntimeEvidenceProjection::Create(
      RuntimeEvidenceBundleState::kComplete,
      RuntimeEvidenceReplayState::kPassed, RuntimeHardwareQualificationState::kSupported,
      evidence_roots).value();
}

RuntimeProfileRoots runtime_roots() {
  const auto projection = runtime_evidence_projection();
  return {runtime_root(1), runtime_root(2), runtime_root(3),
          projection.projection_root(), runtime_root(5), runtime_root(6),
          runtime_root(7)};
}

ProfileSignature runtime_signature() {
  return {"release-1",
          std::vector<std::byte>(kEd25519SignatureBytes, std::byte{0x5a})};
}

class AcceptingRuntimeVerifier final : public ProfileEd25519Verifier {
 public:
  Status verify(std::string_view, std::span<const std::byte>,
                std::span<const std::byte>,
                std::span<const std::byte>) override {
    return Status::Ok();
  }
};

class RuntimeDeviceProbe final : public RuntimeProfileDeviceProbe {
 public:
  explicit RuntimeDeviceProbe(RuntimeProfileGpuFamily family,
                              std::uint64_t total_memory_bytes = 0)
      : family_(family), total_memory_bytes_(total_memory_bytes) {}
  Result<RuntimeProfileDeviceObservation> observe(
      std::int32_t ordinal) override {
    RuntimeProfileDeviceObservation value{};
    value.ordinal = ordinal;
    value.uuid[0] = static_cast<std::byte>(ordinal + 1);
    if (family_ == RuntimeProfileGpuFamily::kRtx4090D24GiB) {
      value.name = "NVIDIA GeForce RTX 4090 D";
      value.compute_major = 8;
      value.compute_minor = 9;
      value.total_memory_bytes = total_memory_bytes_ == 0
                                     ? 24'000'000'000ULL
                                     : total_memory_bytes_;
    } else {
      value.name = "NVIDIA H100 PCIe";
      value.compute_major = 9;
      value.compute_minor = 0;
      value.total_memory_bytes = total_memory_bytes_ == 0
                                     ? 80'000'000'000ULL
                                     : total_memory_bytes_;
    }
    return value;
  }
 private:
  RuntimeProfileGpuFamily family_;
  std::uint64_t total_memory_bytes_ = 0;
};

class RuntimeLeaseOwner final : public RuntimeProfileReferenceLeaseOwner {};

class RuntimeLeaseProbe final : public RuntimeProfileReferenceLeaseProbe {
 public:
  Result<RuntimeProfileReferenceLeaseObservation> observe(
      const RuntimeProfileReferenceDescriptor& descriptor) override {
    return RuntimeProfileReferenceLeaseObservation{
        descriptor.role, descriptor.exact_bytes, descriptor.object_root,
        true, true, std::make_shared<RuntimeLeaseOwner>()};
  }
};

class RuntimeAuthorityOwner final : public RuntimeProfileAuthorityLeaseOwner {};

class RuntimeAuthorityProbe final : public RuntimeProfileAuthorityLeaseProbe {
 public:
  Result<RuntimeProfileAuthorityLeaseObservation> observe(
      const RuntimeProfileAuthorityDescriptor& descriptor) override {
    return RuntimeProfileAuthorityLeaseObservation{
        descriptor.role, descriptor.exact_bytes, descriptor.content_root,
        true, true, std::make_shared<RuntimeAuthorityOwner>()};
  }
};

RuntimeProfileReferenceClosure runtime_closure() {
  const auto roots = runtime_roots();
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

TEST(RuntimeProfilePayloadTest, RoundTripsClosedQwenAndDeepSeekAxes) {
  auto qwen = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kQwen3_0_6B, RuntimeProfileWeightFormat::kXingInt4,
      RuntimeProfileGpuFamily::kRtx4090D24GiB, 1, false,
      RuntimeProfileResidency::kFullResident,
      RuntimeProfileEvidenceState::kHardwareEvidenceOpen, runtime_roots());
  ASSERT_TRUE(qwen.ok());
  auto parsed = parse_runtime_profile_payload(qwen->canonical_bytes());
  ASSERT_TRUE(parsed.ok());
  EXPECT_EQ(parsed->model(), RuntimeProfileModel::kQwen3_0_6B);
  EXPECT_EQ(parsed->weight_format(), RuntimeProfileWeightFormat::kXingInt4);
  EXPECT_EQ(parsed->world_size(), 1);
  EXPECT_EQ(parsed->evidence_state(),
            RuntimeProfileEvidenceState::kHardwareEvidenceOpen);

  auto qwen_h100 = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kQwen3_0_6B, RuntimeProfileWeightFormat::kBf16,
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 1, false,
      RuntimeProfileResidency::kFullResident,
      RuntimeProfileEvidenceState::kHardwareEvidenceOpen, runtime_roots());
  ASSERT_TRUE(qwen_h100.ok());
  auto parsed_qwen_h100 =
      parse_runtime_profile_payload(qwen_h100->canonical_bytes());
  ASSERT_TRUE(parsed_qwen_h100.ok());
  EXPECT_EQ(parsed_qwen_h100->gpu_family(),
            RuntimeProfileGpuFamily::kH100Pcie80GiB);
  EXPECT_EQ(parsed_qwen_h100->weight_format(),
            RuntimeProfileWeightFormat::kBf16);

  auto deepseek = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative,
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 4, true,
      RuntimeProfileResidency::kFullResident,
      RuntimeProfileEvidenceState::kCorrectnessSupported, runtime_roots());
  ASSERT_TRUE(deepseek.ok());
  EXPECT_TRUE(parse_runtime_profile_payload(deepseek->canonical_bytes()).ok());
}

TEST(RuntimeProfilePayloadTest, RejectsImpossibleFamilyAxesAndOpenEncoding) {
  EXPECT_FALSE(RuntimeProfilePayload::Create(
                   RuntimeProfileModel::kQwen3_0_6B,
                   RuntimeProfileWeightFormat::kBf16,
                   RuntimeProfileGpuFamily::kH100Pcie80GiB, 2, false,
                   RuntimeProfileResidency::kHostSpill,
                   RuntimeProfileEvidenceState::kHardwareEvidenceOpen,
                   runtime_roots())
                   .ok());
  EXPECT_FALSE(RuntimeProfilePayload::Create(
                   RuntimeProfileModel::kDeepSeekV4Flash0731,
                   RuntimeProfileWeightFormat::kDeepSeekNative,
                   RuntimeProfileGpuFamily::kRtx4090D24GiB, 3, true,
                   RuntimeProfileResidency::kHostSpill,
                   RuntimeProfileEvidenceState::kHardwareEvidenceOpen,
                   runtime_roots())
                   .ok());
  auto valid = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kQwen3_0_6B, RuntimeProfileWeightFormat::kBf16,
      RuntimeProfileGpuFamily::kRtx4090D24GiB, 1, false,
      RuntimeProfileResidency::kFullResident,
      RuntimeProfileEvidenceState::kHardwareEvidenceOpen, runtime_roots());
  ASSERT_TRUE(valid.ok());
  auto bytes = std::vector<std::byte>(valid->canonical_bytes().begin(),
                                      valid->canonical_bytes().end());
  bytes.push_back(std::byte{0});
  EXPECT_FALSE(parse_runtime_profile_payload(bytes).ok());
  bytes.pop_back();
  std::fill(bytes.end() - 32, bytes.end(), std::byte{0});
  EXPECT_FALSE(parse_runtime_profile_payload(bytes).ok());
}

TEST(RuntimeProfilePayloadTest, BindsOnlyTheAuthorityVerifiedEnvelope) {
  auto payload = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kQwen3_0_6B, RuntimeProfileWeightFormat::kBf16,
      RuntimeProfileGpuFamily::kRtx4090D24GiB, 1, false,
      RuntimeProfileResidency::kFullResident,
      RuntimeProfileEvidenceState::kHardwareEvidenceOpen, runtime_roots());
  ASSERT_TRUE(payload.ok());
  auto envelope = SignedProfileEnvelope::Create(
      "qwen-exact-1", "r1", 9, payload->canonical_bytes(), runtime_root(9),
      {runtime_signature()});
  ASSERT_TRUE(envelope.ok());
  auto catalog = SignedProfileCatalog::Create(
      9, {{"qwen-exact-1", "r1", envelope->envelope_root(),
           envelope->envelope_bytes()}},
      {runtime_signature()});
  ASSERT_TRUE(catalog.ok());
  ProfileTrustKey key{};
  key.key_id = "release-1";
  key.role = ProfileSignerRole::kRelease;
  key.public_key.fill(std::byte{0x22});
  auto policy = ProfileTrustPolicy::Create(9, catalog->catalog_root(), {key});
  ASSERT_TRUE(policy.ok());
  AcceptingRuntimeVerifier verifier;
  auto authority = verify_profile_authority(
      *policy, *catalog, *envelope, "qwen-exact-1", "r1", verifier);
  ASSERT_TRUE(authority.ok());

  auto bound = bind_verified_runtime_profile(*authority, *envelope);
  ASSERT_TRUE(bound.ok());
  EXPECT_EQ(bound->profile_id(), "qwen-exact-1");
  EXPECT_EQ(bound->payload().model(), RuntimeProfileModel::kQwen3_0_6B);

  auto splice = SignedProfileEnvelope::Create(
      "other-profile", "r1", 9, payload->canonical_bytes(), runtime_root(9),
      {runtime_signature()});
  ASSERT_TRUE(splice.ok());
  EXPECT_FALSE(bind_verified_runtime_profile(*authority, *splice).ok());
}

TEST(RuntimeProfilePayloadTest, HardwareOpenCannotIssueEngineAdmission) {
  auto open = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kQwen3_0_6B, RuntimeProfileWeightFormat::kBf16,
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 1, false,
      RuntimeProfileResidency::kFullResident,
      RuntimeProfileEvidenceState::kHardwareEvidenceOpen, runtime_roots());
  auto supported = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative,
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 4, true,
      RuntimeProfileResidency::kHostSpill,
      RuntimeProfileEvidenceState::kCorrectnessSupported, runtime_roots());
  ASSERT_TRUE(open.ok());
  ASSERT_TRUE(supported.ok());

  // The private verified type is exercised through the full authority binding
  // in the preceding test; here the public admission behavior is covered by
  // reusing that construction in a compact local fixture below.
  auto make_verified = [](const RuntimeProfilePayload& payload,
                          std::string profile_id,
                          Sha256Digest reference_root) {
    auto envelope = SignedProfileEnvelope::Create(
        profile_id, "r1", 9, payload.canonical_bytes(), reference_root,
        {runtime_signature()}).value();
    auto graph = verify_sealed_reachable_object_dag(
        {{"schema_v1", runtime_root(20), 1, 1, true, {}}},
        {runtime_root(20)}).value();
    auto catalog = SignedProfileCatalog::CreateGraphBound(
        9, {graph.snapshot_root(), graph.node_count(), graph.edge_count()},
        {{profile_id, "r1", envelope.envelope_root(),
             envelope.envelope_bytes()}},
        {runtime_signature()}).value();
    ProfileTrustKey key{};
    key.key_id = "release-1";
    key.role = ProfileSignerRole::kRelease;
    key.public_key.fill(std::byte{0x22});
    auto policy = ProfileTrustPolicy::Create(9, catalog.catalog_root(), {key})
                      .value();
    AcceptingRuntimeVerifier verifier;
    auto authority = verify_profile_authority(
        policy, catalog, envelope, profile_id, "r1", verifier).value();
    return bind_verified_runtime_profile(authority, envelope).value();
  };
  auto open_profile = make_verified(*open, "qwen-open", runtime_root(9));
  RuntimeDeviceProbe qwen_probe(RuntimeProfileGpuFamily::kH100Pcie80GiB);
  const std::array<std::int32_t, 1> qwen_ordinals{0};
  auto qwen_devices = verify_runtime_profile_devices(
      open_profile, qwen_ordinals, qwen_probe);
  ASSERT_TRUE(qwen_devices.ok());
  RuntimeDeviceProbe oversized_qwen_probe(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 90'000'000'000ULL);
  EXPECT_FALSE(verify_runtime_profile_devices(
                   open_profile, qwen_ordinals, oversized_qwen_probe)
                   .ok());
  EXPECT_FALSE(admit_runtime_engine(
                   open_profile, *qwen_devices,
                   RuntimeProfileModel::kQwen3_0_6B,
                   RuntimeProfileWeightFormat::kBf16)
                   .ok());

  auto closure = runtime_closure();
  auto supported_profile = make_verified(
      *supported, "deepseek-supported", closure.closure_root());
  RuntimeDeviceProbe h100_probe(RuntimeProfileGpuFamily::kH100Pcie80GiB);
  const std::array<std::int32_t, 4> h100_ordinals{0, 1, 2, 3};
  auto h100_devices = verify_runtime_profile_devices(
      supported_profile, h100_ordinals, h100_probe);
  ASSERT_TRUE(h100_devices.ok());
  auto admitted = admit_runtime_engine(
      supported_profile, *h100_devices,
      RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative);
  ASSERT_TRUE(admitted.ok());
  EXPECT_EQ(admitted->profile_id(), "deepseek-supported");
  EXPECT_EQ(admitted->model(), RuntimeProfileModel::kDeepSeekV4Flash0731);
  EXPECT_EQ(admitted->weight_format(),
            RuntimeProfileWeightFormat::kDeepSeekNative);
  EXPECT_TRUE(admitted->dspark_enabled());
  EXPECT_EQ(admitted->residency(), RuntimeProfileResidency::kHostSpill);
  EXPECT_EQ(admitted->device_ordinals().size(), 4U);
  EXPECT_EQ(admitted->device_observation_root(),
            h100_devices->observation_root());
  EXPECT_EQ(admitted->capacity_template_exact_bytes(), 0U);
  EXPECT_FALSE(admitted->references_retained());
  auto evidence_bound = bind_runtime_engine_evidence_projection(
      std::move(*admitted), runtime_evidence_projection());
  ASSERT_TRUE(evidence_bound.ok());
  RuntimeLeaseProbe lease_probe;
  auto leases = verify_runtime_profile_reference_leases(closure, lease_probe);
  ASSERT_TRUE(leases.ok());
  auto retained = retain_runtime_engine_reference_leases(
      std::move(*evidence_bound), std::move(*leases));
  ASSERT_TRUE(retained.ok());
  EXPECT_TRUE(retained->references_retained());
  EXPECT_TRUE(retained->production_references_leased());
  EXPECT_EQ(retained->capacity_template_exact_bytes(), 2U);
  EXPECT_FALSE(retained->production_authority_leased());
  EXPECT_FALSE(retained->production_ready());
  EXPECT_FALSE(retained->production_compute_ready());
  auto retained_envelope = SignedProfileEnvelope::Create(
      "deepseek-supported", "r1", 9, supported->canonical_bytes(),
      closure.closure_root(), {runtime_signature()}).value();
  auto retained_graph = verify_sealed_reachable_object_dag(
      {{"schema_v1", runtime_root(20), 1, 1, true, {}}},
      {runtime_root(20)}).value();
  auto retained_catalog = SignedProfileCatalog::CreateGraphBound(
      9, {retained_graph.snapshot_root(), retained_graph.node_count(),
          retained_graph.edge_count()},
      {{"deepseek-supported", "r1", retained_envelope.envelope_root(),
           retained_envelope.envelope_bytes()}},
      {runtime_signature()}).value();
  ProfileTrustKey retained_key{};
  retained_key.key_id = "release-1";
  retained_key.role = ProfileSignerRole::kRelease;
  retained_key.public_key.fill(std::byte{0x22});
  auto retained_policy = ProfileTrustPolicy::Create(
      9, retained_catalog.catalog_root(), {retained_key}).value();
  RuntimeAuthorityProbe authority_probe;
  auto authority_leases = verify_runtime_profile_authority_leases(
      retained_policy, retained_catalog, retained_envelope, closure,
      retained_graph, authority_probe);
  ASSERT_TRUE(authority_leases.ok());
  auto production = retain_runtime_engine_authority_leases(
      std::move(*retained), std::move(*authority_leases));
  ASSERT_TRUE(production.ok()) << production.status().message();
  EXPECT_TRUE(production->production_authority_leased());
  EXPECT_TRUE(production->production_ready());
  EXPECT_TRUE(production->production_compute_ready());
  auto inherited_fds = RuntimeProfileInheritedFdManifest::Create(
      {3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14}).value();
  std::array<std::byte, kRuntimeReadinessNonceBytes> readiness_nonce{};
  readiness_nonce.fill(std::byte{0x33});
  const auto authority_snapshot_root = runtime_root(10);
  auto bootstrap = RuntimeProfileSupervisorBootstrapManifest::Create(
      std::move(inherited_fds), 17, retained_envelope.envelope_root(),
      authority_snapshot_root, readiness_nonce);
  ASSERT_TRUE(bootstrap.ok());
  auto readiness = issue_runtime_profile_readiness_receipt(
      *production, *bootstrap, readiness_nonce, authority_snapshot_root,
      runtime_root(11));
  ASSERT_TRUE(readiness.ok()) << readiness.status().message();
  EXPECT_EQ(readiness->deployment_generation(), 17U);
  EXPECT_EQ(readiness->authority_snapshot_root(), authority_snapshot_root);
  EXPECT_EQ(readiness->bootstrap_manifest_root(), bootstrap->manifest_root());
  EXPECT_EQ(readiness->device_observation_root(),
            production->device_observation_root());
  EXPECT_EQ(readiness->policy_digest(), retained_policy.policy_digest());
  EXPECT_EQ(readiness->catalog_root(), retained_catalog.catalog_root());
  EXPECT_EQ(readiness->envelope_root(), retained_envelope.envelope_root());
  EXPECT_EQ(readiness->reference_closure_root(), closure.closure_root());
  EXPECT_EQ(readiness->graph_snapshot_root(), retained_graph.snapshot_root());
  auto bad_nonce = readiness_nonce;
  bad_nonce[0] = std::byte{0x34};
  EXPECT_FALSE(issue_runtime_profile_readiness_receipt(
      *production, *bootstrap, bad_nonce, authority_snapshot_root,
      runtime_root(11)).ok());
  EXPECT_FALSE(issue_runtime_profile_readiness_receipt(
      *production, *bootstrap, readiness_nonce, runtime_root(12),
      runtime_root(11)).ok());
  auto placement = CatalogPlacementTransaction::Create(
      runtime_root(13), runtime_root(14), *bootstrap,
      CatalogPlacementDeadlinePolicy::Create(10, 20, 30, 40, 50).value(),
      100);
  ASSERT_TRUE(placement.ok());
  EXPECT_EQ(placement->state(), CatalogPlacementState::kVerifying);
  const auto placement_snapshot = placement->recovery_snapshot();
  EXPECT_EQ(placement_snapshot.transaction_id, runtime_root(13));
  EXPECT_EQ(placement_snapshot.absolute_deadlines[0], 110U);
  EXPECT_EQ(placement_snapshot.state, CatalogPlacementState::kVerifying);
  auto encoded_snapshot = encode_catalog_placement_recovery_snapshot(placement_snapshot);
  ASSERT_TRUE(encoded_snapshot.ok());
  auto decoded_snapshot = parse_catalog_placement_recovery_snapshot(*encoded_snapshot);
  ASSERT_TRUE(decoded_snapshot.ok());
  EXPECT_EQ(decoded_snapshot->transaction_id, placement_snapshot.transaction_id);
  EXPECT_EQ(decoded_snapshot->absolute_deadlines, placement_snapshot.absolute_deadlines);
  auto routed_snapshot = placement_snapshot;
  routed_snapshot.state = CatalogPlacementState::kRouted;
  EXPECT_FALSE(encode_catalog_placement_recovery_snapshot(routed_snapshot).ok());
  (*encoded_snapshot)[8] ^= std::byte{1};
  EXPECT_FALSE(parse_catalog_placement_recovery_snapshot(*encoded_snapshot).ok());
  (*encoded_snapshot)[8] ^= std::byte{1};
  encoded_snapshot->pop_back();
  EXPECT_FALSE(parse_catalog_placement_recovery_snapshot(*encoded_snapshot).ok());
  const auto staging = std::filesystem::temp_directory_path() /
                       "pih-placement-recovery.staging";
  const auto published = std::filesystem::temp_directory_path() /
                         "pih-placement-recovery.record";
  std::filesystem::remove(staging);
  std::filesystem::remove(published);
  auto publication = publish_catalog_placement_recovery_snapshot(
      staging, published, placement_snapshot);
  ASSERT_TRUE(publication.ok());
  EXPECT_TRUE(std::filesystem::exists(published));
  auto loaded_snapshot = load_catalog_placement_recovery_snapshot(published);
  ASSERT_TRUE(loaded_snapshot.ok());
  EXPECT_EQ(loaded_snapshot->transaction_id, placement_snapshot.transaction_id);
  EXPECT_EQ(classify_catalog_placement_recovery(*loaded_snapshot),
            CatalogPlacementRecoveryAction::kRequireReconciliation);
  loaded_snapshot->state = CatalogPlacementState::kWithdrawn;
  EXPECT_EQ(classify_catalog_placement_recovery(*loaded_snapshot),
            CatalogPlacementRecoveryAction::kReleased);
  loaded_snapshot->state = CatalogPlacementState::kQuarantined;
  EXPECT_EQ(classify_catalog_placement_recovery(*loaded_snapshot),
            CatalogPlacementRecoveryAction::kQuarantined);
  {
    std::fstream mutation(published, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(mutation.good());
    mutation.seekp(8);
    mutation.put('\x7f');
  }
  EXPECT_FALSE(load_catalog_placement_recovery_snapshot(published).ok());
  EXPECT_FALSE(publish_catalog_placement_recovery_snapshot(
      staging, published, placement_snapshot).ok());
  std::filesystem::remove(staging);
  std::filesystem::remove(published);
  EXPECT_FALSE(placement->bind_readiness(*readiness).ok());
  ASSERT_TRUE(placement->mark_starting().ok());
  auto foreign_fds = RuntimeProfileInheritedFdManifest::Create(
      {3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14}).value();
  auto foreign_nonce = readiness_nonce;
  foreign_nonce[0] = std::byte{0x44};
  auto foreign_bootstrap = RuntimeProfileSupervisorBootstrapManifest::Create(
      std::move(foreign_fds), 17, retained_envelope.envelope_root(),
      authority_snapshot_root, foreign_nonce).value();
  auto foreign_readiness = issue_runtime_profile_readiness_receipt(
      *production, foreign_bootstrap, foreign_nonce, authority_snapshot_root,
      runtime_root(11)).value();
  EXPECT_FALSE(placement->bind_readiness(foreign_readiness).ok());
  ASSERT_TRUE(placement->bind_readiness(*readiness).ok());
  EXPECT_EQ(placement->state(), CatalogPlacementState::kRouteCommitting);
  EXPECT_FALSE(placement->commit_route(runtime_root(12),
                                       readiness->receipt_root()).ok());
  EXPECT_FALSE(placement->commit_route(authority_snapshot_root,
                                       runtime_root(12)).ok());
  ASSERT_TRUE(placement->commit_route(authority_snapshot_root,
                                      readiness->receipt_root()).ok());
  EXPECT_EQ(placement->state(), CatalogPlacementState::kRouted);
  EXPECT_FALSE(placement->fail().ok());
  EXPECT_FALSE(placement->acknowledge_route_withdrawal(
      readiness->receipt_root()).ok());
  ASSERT_TRUE(placement->begin_route_withdrawal().ok());
  EXPECT_EQ(placement->state(), CatalogPlacementState::kRouteWithdrawing);
  EXPECT_FALSE(placement->fail().ok());
  EXPECT_FALSE(placement->acknowledge_route_withdrawal(runtime_root(12)).ok());
  ASSERT_TRUE(placement->acknowledge_route_withdrawal(
      readiness->receipt_root()).ok());
  EXPECT_EQ(placement->state(), CatalogPlacementState::kWithdrawn);
  EXPECT_FALSE(placement->abort().ok());
  EXPECT_FALSE(admit_runtime_engine(
                   supported_profile, *h100_devices,
                   RuntimeProfileModel::kQwen3_0_6B,
                   RuntimeProfileWeightFormat::kDeepSeekNative)
                   .ok());
}

TEST(RuntimeProfilePayloadTest, DeviceGateRejectsCountClassAndDuplicateUuid) {
  auto payload = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative,
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 2, true,
      RuntimeProfileResidency::kHostSpill,
      RuntimeProfileEvidenceState::kCorrectnessSupported, runtime_roots());
  ASSERT_TRUE(payload.ok());
  // Full authority fixture is intentionally reused through a local equivalent.
  auto envelope = SignedProfileEnvelope::Create(
      "deepseek-h100-2", "r1", 9, payload->canonical_bytes(), runtime_root(9),
      {runtime_signature()}).value();
  auto catalog = SignedProfileCatalog::Create(
      9, {{"deepseek-h100-2", "r1", envelope.envelope_root(),
           envelope.envelope_bytes()}}, {runtime_signature()}).value();
  ProfileTrustKey key{};
  key.key_id = "release-1";
  key.role = ProfileSignerRole::kRelease;
  key.public_key.fill(std::byte{0x22});
  auto policy = ProfileTrustPolicy::Create(9, catalog.catalog_root(), {key}).value();
  AcceptingRuntimeVerifier verifier;
  auto authority = verify_profile_authority(
      policy, catalog, envelope, "deepseek-h100-2", "r1", verifier).value();
  auto profile = bind_verified_runtime_profile(authority, envelope).value();

  RuntimeDeviceProbe wrong_class(RuntimeProfileGpuFamily::kRtx4090D24GiB);
  RuntimeDeviceProbe right_class(RuntimeProfileGpuFamily::kH100Pcie80GiB);
  const std::array<std::int32_t, 1> one{0};
  const std::array<std::int32_t, 2> duplicate{0, 0};
  EXPECT_FALSE(verify_runtime_profile_devices(profile, one, right_class).ok());
  EXPECT_FALSE(verify_runtime_profile_devices(profile, duplicate, right_class).ok());
  const std::array<std::int32_t, 2> two{0, 1};
  EXPECT_FALSE(verify_runtime_profile_devices(profile, two, wrong_class).ok());
  RuntimeDeviceProbe h100_nvl_sized(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 94'000'000'000ULL);
  EXPECT_FALSE(
      verify_runtime_profile_devices(profile, two, h100_nvl_sized).ok());
  RuntimeDeviceProbe h100_upper_boundary(
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 90'000'000'000ULL);
  EXPECT_FALSE(
      verify_runtime_profile_devices(profile, two, h100_upper_boundary).ok());
}

}  // namespace
}  // namespace pih
