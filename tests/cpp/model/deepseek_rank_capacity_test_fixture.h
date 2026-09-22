#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pih/model/deepseek_rank_capacity_plan_instance.h"
#include "pih/model/profile_authority_codec.h"
#include "pih/model/runtime_evidence_projection.h"
#include "pih/model/runtime_profile_authority_lease.h"
#include "pih/model/runtime_profile_device_gate.h"
#include "pih/model/runtime_profile_reference_closure.h"
#include "pih/model/runtime_profile_reference_lease.h"
#include "pih/model/sealed_reachable_object_dag.h"

namespace pih::test_fixture {

inline Sha256Digest rank_capacity_digest(std::uint8_t seed) {
  Sha256Digest value{};
  value.bytes.fill(static_cast<std::byte>(seed));
  return value;
}

inline ProfileSignature rank_capacity_signature() {
  return {"release-1",
          std::vector<std::byte>(kEd25519SignatureBytes, std::byte{0x5a})};
}

class RankCapacityAcceptingVerifier final : public ProfileEd25519Verifier {
 public:
  Status verify(std::string_view, std::span<const std::byte>,
                std::span<const std::byte>,
                std::span<const std::byte>) override {
    return Status::Ok();
  }
};

class RankCapacityDeviceProbe final : public RuntimeProfileDeviceProbe {
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

class RankCapacityReferenceOwner final
    : public RuntimeProfileReferenceLeaseOwner {};

class RankCapacityReferenceProbe final
    : public RuntimeProfileReferenceLeaseProbe {
 public:
  Result<RuntimeProfileReferenceLeaseObservation> observe(
      const RuntimeProfileReferenceDescriptor& descriptor) override {
    return RuntimeProfileReferenceLeaseObservation{
        descriptor.role, descriptor.exact_bytes, descriptor.object_root,
        true, true, std::make_shared<RankCapacityReferenceOwner>()};
  }
};

class RankCapacityAuthorityOwner final
    : public RuntimeProfileAuthorityLeaseOwner {};

class RankCapacityAuthorityProbe final
    : public RuntimeProfileAuthorityLeaseProbe {
 public:
  Result<RuntimeProfileAuthorityLeaseObservation> observe(
      const RuntimeProfileAuthorityDescriptor& descriptor) override {
    return RuntimeProfileAuthorityLeaseObservation{
        descriptor.role, descriptor.exact_bytes, descriptor.content_root,
        true, true, std::make_shared<RankCapacityAuthorityOwner>()};
  }
};

inline RuntimeEvidenceProjection rank_capacity_evidence() {
  return RuntimeEvidenceProjection::Create(
      RuntimeEvidenceBundleState::kComplete,
      RuntimeEvidenceReplayState::kPassed,
      RuntimeHardwareQualificationState::kSupported,
      {rank_capacity_digest(30), rank_capacity_digest(31),
       rank_capacity_digest(32), rank_capacity_digest(33),
       rank_capacity_digest(34)})
      .value();
}

inline RuntimeProfileRoots rank_capacity_profile_roots(
    const RuntimeEvidenceProjection& evidence) {
  return {rank_capacity_digest(1), rank_capacity_digest(2),
          rank_capacity_digest(3), evidence.projection_root(),
          rank_capacity_digest(5), rank_capacity_digest(6),
          rank_capacity_digest(7)};
}

inline RuntimeProfileReferenceClosure rank_capacity_closure(
    const RuntimeProfileRoots& roots) {
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

inline RuntimeEngineAdmission rank_capacity_admission(
    std::uint8_t world_size, bool production_ready = true,
    bool bind_evidence = true,
    std::span<const std::int32_t> admitted_ordinals = {},
    const RuntimeEvidenceProjection* supplied_evidence = nullptr,
    bool dspark_enabled = false,
    RuntimeProfileResidency residency = RuntimeProfileResidency::kHostSpill) {
  auto default_evidence = rank_capacity_evidence();
  const auto& evidence = supplied_evidence == nullptr
                             ? default_evidence
                             : *supplied_evidence;
  const auto roots = rank_capacity_profile_roots(evidence);
  auto closure = rank_capacity_closure(roots);
  auto payload = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative,
      RuntimeProfileGpuFamily::kH100Pcie80GiB, world_size, dspark_enabled,
      residency,
      RuntimeProfileEvidenceState::kCorrectnessSupported, roots).value();
  auto envelope = SignedProfileEnvelope::Create(
      "deepseek-rank-test", "r1", 9, payload.canonical_bytes(),
      closure.closure_root(), {rank_capacity_signature()}).value();
  auto graph = verify_sealed_reachable_object_dag(
      {{"schema_v1", rank_capacity_digest(40), 1, 1, true, {}}},
      {rank_capacity_digest(40)}).value();
  auto catalog = SignedProfileCatalog::CreateGraphBound(
      9, {graph.snapshot_root(), graph.node_count(), graph.edge_count()},
      {{"deepseek-rank-test", "r1", envelope.envelope_root(),
        envelope.envelope_bytes()}},
      {rank_capacity_signature()}).value();
  ProfileTrustKey key{};
  key.key_id = "release-1";
  key.role = ProfileSignerRole::kRelease;
  key.public_key.fill(std::byte{0x22});
  auto policy = ProfileTrustPolicy::Create(
      9, catalog.catalog_root(), {key}).value();
  RankCapacityAcceptingVerifier verifier;
  auto authority = verify_profile_authority(
      policy, catalog, envelope, "deepseek-rank-test", "r1", verifier)
                       .value();
  auto profile = bind_verified_runtime_profile(authority, envelope).value();
  std::vector<std::int32_t> ordinals;
  if (admitted_ordinals.empty()) {
    ordinals.resize(world_size);
    for (std::uint8_t rank = 0; rank < world_size; ++rank) {
      ordinals[rank] = rank;
    }
  } else {
    ordinals.assign(admitted_ordinals.begin(), admitted_ordinals.end());
  }
  RankCapacityDeviceProbe device_probe;
  auto devices = verify_runtime_profile_devices(
      profile, ordinals, device_probe).value();
  auto admission = admit_runtime_engine(
      profile, devices, RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative).value();
  if (bind_evidence) {
    admission = bind_runtime_engine_evidence_projection(
        std::move(admission), evidence).value();
  }
  if (!production_ready) return admission;

  RankCapacityReferenceProbe reference_probe;
  auto reference_leases = verify_runtime_profile_reference_leases(
      closure, reference_probe).value();
  admission = retain_runtime_engine_reference_leases(
      std::move(admission), std::move(reference_leases)).value();
  RankCapacityAuthorityProbe authority_probe;
  auto authority_leases = verify_runtime_profile_authority_leases(
      policy, catalog, envelope, closure, graph, authority_probe).value();
  return retain_runtime_engine_authority_leases(
      std::move(admission), std::move(authority_leases)).value();
}

inline RuntimeProfileSupervisorBootstrapManifest rank_capacity_bootstrap(
    const RuntimeEngineAdmission& admission,
    std::uint64_t deployment_generation = 17,
    std::uint8_t authority_snapshot_seed = 60) {
  auto fds = RuntimeProfileInheritedFdManifest::Create(
      {3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14}).value();
  std::array<std::byte, kRuntimeReadinessNonceBytes> nonce{};
  nonce.fill(std::byte{0x33});
  return RuntimeProfileSupervisorBootstrapManifest::Create(
      std::move(fds), deployment_generation, admission.envelope_root(),
      rank_capacity_digest(authority_snapshot_seed), nonce).value();
}

inline DeepSeekRankCapacityPlanInstance rank_capacity_instance(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& resource_plan,
    std::uint64_t controller_process_identity = 90,
    std::uint64_t deployment_generation = 17) {
  std::vector<std::int32_t> ordinals;
  ordinals.reserve(manifests.size());
  for (const auto& manifest : manifests) {
    ordinals.push_back(manifest.startup_device_ordinal);
  }
  auto admission = rank_capacity_admission(
      static_cast<std::uint8_t>(manifests.size()), true, true, ordinals);
  auto bootstrap = rank_capacity_bootstrap(
      admission, deployment_generation);
  return DeepSeekRankCapacityPlanInstance::Compile(
      admission, bootstrap, manifests, resource_plan,
      controller_process_identity).value();
}

}  // namespace pih::test_fixture
