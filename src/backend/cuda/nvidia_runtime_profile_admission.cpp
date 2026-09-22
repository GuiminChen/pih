#include "pih/backend/cuda/nvidia_runtime_profile_admission.h"

#include "pih/backend/cuda/nvidia_runtime_profile_device_probe.h"
#include "pih/model/openssl_profile_ed25519_verifier.h"
#include "pih/model/profile_authority_codec.h"
#include "pih/model/runtime_profile_device_gate.h"
#include "pih/model/runtime_profile_authority_lease.h"
#include "pih/model/runtime_profile_reference_closure.h"
#include "pih/model/runtime_evidence_projection.h"
#if defined(__linux__)
#include "pih/platform/linux/linux_runtime_profile_authority_lease_probe.h"
#include "pih/platform/linux/linux_runtime_profile_reference_lease_probe.h"
#endif

namespace pih {

Result<RuntimeEngineAdmission> admit_nvidia_runtime_profile(
    std::span<const std::byte> trust_policy_bytes,
    std::span<const std::byte> catalog_bytes,
    std::span<const std::byte> envelope_bytes,
    std::span<const std::byte> reference_closure_bytes,
    std::span<const std::vector<std::byte>> ordered_reference_objects,
    std::string_view profile_id,
    std::string_view revision,
    std::span<const std::int32_t> ordered_device_ordinals,
    RuntimeProfileModel expected_model,
    RuntimeProfileWeightFormat expected_weight_format) {
  auto policy = parse_profile_trust_policy(trust_policy_bytes);
  if (!policy.ok()) return policy.status();
  auto catalog = parse_signed_profile_catalog(catalog_bytes);
  if (!catalog.ok()) return catalog.status();
  auto envelope = parse_signed_profile_envelope(envelope_bytes);
  if (!envelope.ok()) return envelope.status();
  OpenSslProfileEd25519Verifier signature_verifier;
  auto authority = verify_profile_authority(
      *policy, *catalog, *envelope, profile_id, revision,
      signature_verifier);
  if (!authority.ok()) return authority.status();
  auto profile = bind_verified_runtime_profile(*authority, *envelope);
  if (!profile.ok()) return profile.status();
  auto reference_closure = parse_runtime_profile_reference_closure(
      reference_closure_bytes);
  if (!reference_closure.ok()) return reference_closure.status();
  auto closure_status = verify_runtime_profile_reference_closure(
      *profile, *envelope, *reference_closure);
  if (!closure_status.ok()) return closure_status;
  auto object_status = verify_runtime_profile_reference_objects(
      *reference_closure, ordered_reference_objects);
  if (!object_status.ok()) return object_status;
  const auto release_index = static_cast<std::size_t>(
      RuntimeProfileReferenceRole::kReleaseEvidence) - 1;
  auto evidence_projection = parse_runtime_evidence_projection(
      ordered_reference_objects[release_index]);
  if (!evidence_projection.ok()) return evidence_projection.status();
  auto evidence_status = verify_runtime_evidence_projection(
      *profile, *evidence_projection);
  if (!evidence_status.ok()) return evidence_status;
  NvidiaRuntimeProfileDeviceProbe device_probe;
  auto devices = verify_runtime_profile_devices(
      *profile, ordered_device_ordinals, device_probe);
  if (!devices.ok()) return devices.status();
  auto admitted = admit_runtime_engine(*profile, *devices, expected_model,
                                       expected_weight_format);
  if (!admitted.ok()) return admitted.status();
  auto evidence_bound = bind_runtime_engine_evidence_projection(
      std::move(*admitted), *evidence_projection);
  if (!evidence_bound.ok()) return evidence_bound.status();
  return retain_runtime_engine_reference_objects(
      std::move(*evidence_bound), *reference_closure,
      std::vector<std::vector<std::byte>>(ordered_reference_objects.begin(),
                                          ordered_reference_objects.end()));
}

#if defined(__linux__)
Result<RuntimeEngineAdmission> admit_nvidia_runtime_profile_from_leases(
    std::span<const std::byte> trust_policy_bytes,
    std::span<const std::byte> catalog_bytes,
    std::span<const std::byte> envelope_bytes,
    std::span<const std::byte> reference_closure_bytes,
    std::span<const std::byte> graph_manifest_bytes,
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap_manifest,
    std::string_view profile_id,
    std::string_view revision,
    std::span<const std::int32_t> ordered_device_ordinals,
    RuntimeProfileModel expected_model,
    RuntimeProfileWeightFormat expected_weight_format) {
  auto policy = parse_profile_trust_policy(trust_policy_bytes);
  if (!policy.ok()) return policy.status();
  auto catalog = parse_signed_profile_catalog(catalog_bytes);
  if (!catalog.ok()) return catalog.status();
  auto envelope = parse_signed_profile_envelope(envelope_bytes);
  if (!envelope.ok()) return envelope.status();
  if (!(envelope->envelope_root() ==
        bootstrap_manifest.expected_envelope_root())) {
    return Status::FailedPrecondition(
        "runtime envelope differs from supervisor bootstrap manifest");
  }
  OpenSslProfileEd25519Verifier signature_verifier;
  auto authority = verify_profile_authority(
      *policy, *catalog, *envelope, profile_id, revision,
      signature_verifier);
  if (!authority.ok()) return authority.status();
  auto profile = bind_verified_runtime_profile(*authority, *envelope);
  if (!profile.ok()) return profile.status();
  auto reference_closure = parse_runtime_profile_reference_closure(
      reference_closure_bytes);
  if (!reference_closure.ok()) return reference_closure.status();
  auto closure_status = verify_runtime_profile_reference_closure(
      *profile, *envelope, *reference_closure);
  if (!closure_status.ok()) return closure_status;
  auto graph = parse_sealed_reachable_object_dag(graph_manifest_bytes);
  if (!graph.ok()) return graph.status();
  auto authority_probe = LinuxRuntimeProfileAuthorityLeaseProbe::Create(
      bootstrap_manifest.inherited_fds().authority_fds());
  if (!authority_probe.ok()) return authority_probe.status();
  auto authority_leases = verify_runtime_profile_authority_leases(
      *policy, *catalog, *envelope, *reference_closure, *graph,
      *authority_probe);
  if (!authority_leases.ok()) return authority_leases.status();
  auto lease_probe = LinuxRuntimeProfileReferenceLeaseProbe::Create(
      bootstrap_manifest.inherited_fds().reference_fds());
  if (!lease_probe.ok()) return lease_probe.status();
  auto leases = verify_runtime_profile_reference_leases(
      *reference_closure, *lease_probe);
  if (!leases.ok()) return leases.status();
  const auto release_index = static_cast<std::size_t>(
      RuntimeProfileReferenceRole::kReleaseEvidence) - 1;
  auto evidence_bytes = lease_probe->read_bounded_object(
      reference_closure->descriptors()[release_index], 1024 * 1024);
  if (!evidence_bytes.ok()) return evidence_bytes.status();
  auto evidence_projection = parse_runtime_evidence_projection(*evidence_bytes);
  if (!evidence_projection.ok()) return evidence_projection.status();
  auto evidence_status = verify_runtime_evidence_projection(
      *profile, *evidence_projection);
  if (!evidence_status.ok()) return evidence_status;
  NvidiaRuntimeProfileDeviceProbe device_probe;
  auto devices = verify_runtime_profile_devices(
      *profile, ordered_device_ordinals, device_probe);
  if (!devices.ok()) return devices.status();
  auto admitted = admit_runtime_engine(*profile, *devices, expected_model,
                                       expected_weight_format);
  if (!admitted.ok()) return admitted.status();
  auto evidence_bound = bind_runtime_engine_evidence_projection(
      std::move(*admitted), *evidence_projection);
  if (!evidence_bound.ok()) return evidence_bound.status();
  auto reference_retained = retain_runtime_engine_reference_leases(
      std::move(*evidence_bound), std::move(*leases));
  if (!reference_retained.ok()) return reference_retained.status();
  auto production_admission = retain_runtime_engine_authority_leases(
      std::move(*reference_retained), std::move(*authority_leases));
  if (!production_admission.ok()) return production_admission.status();
  if (!production_admission->production_ready()) {
    return Status::FailedPrecondition(
        "runtime production admission did not retain all authority leases");
  }
  return production_admission;
}
#endif

}  // namespace pih
