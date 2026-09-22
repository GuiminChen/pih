#include "pih/model/runtime_profile_authority_lease.h"

#include <array>
#include <utility>

namespace pih {

VerifiedRuntimeProfileAuthorityLeases::VerifiedRuntimeProfileAuthorityLeases(
    Sha256Digest policy_digest, Sha256Digest catalog_root,
    Sha256Digest envelope_root, Sha256Digest reference_closure_root,
    Sha256Digest graph_snapshot_root,
    std::vector<std::shared_ptr<const RuntimeProfileAuthorityLeaseOwner>>
        owners) noexcept
    : policy_digest_(policy_digest), catalog_root_(catalog_root),
      envelope_root_(envelope_root),
      reference_closure_root_(reference_closure_root),
      graph_snapshot_root_(graph_snapshot_root),
      owners_(std::move(owners)) {}

Result<VerifiedRuntimeProfileAuthorityLeases>
verify_runtime_profile_authority_leases(
    const ProfileTrustPolicy& policy,
    const SignedProfileCatalog& catalog,
    const SignedProfileEnvelope& envelope,
    const RuntimeProfileReferenceClosure& closure,
    const SealedReachableObjectDag& graph,
    RuntimeProfileAuthorityLeaseProbe& probe) {
  if (!catalog.graph_bound() ||
      !(catalog.graph_binding()->snapshot_root == graph.snapshot_root()) ||
      catalog.graph_binding()->node_count != graph.node_count() ||
      catalog.graph_binding()->edge_count != graph.edge_count()) {
    return Status::FailedPrecondition(
        "runtime profile catalog differs from digest DAG manifest");
  }
  const std::array<RuntimeProfileAuthorityDescriptor, 5> descriptors{{
      {RuntimeProfileAuthorityRole::kTrustPolicy,
       policy.canonical_object_bytes().size(), policy.policy_digest()},
      {RuntimeProfileAuthorityRole::kCatalog,
       catalog.canonical_object_bytes().size(), catalog.catalog_root()},
      {RuntimeProfileAuthorityRole::kEnvelope,
       envelope.canonical_object_bytes().size(), envelope.envelope_root()},
      {RuntimeProfileAuthorityRole::kReferenceClosure,
       closure.canonical_bytes().size(), closure.closure_root()},
      {RuntimeProfileAuthorityRole::kDigestDagManifest,
       graph.canonical_manifest_bytes().size(), graph.snapshot_root()},
  }};
  std::vector<std::shared_ptr<const RuntimeProfileAuthorityLeaseOwner>> owners;
  owners.reserve(descriptors.size());
  for (const auto& descriptor : descriptors) {
    auto observed = probe.observe(descriptor);
    if (!observed.ok()) return observed.status();
    if (observed->role != descriptor.role ||
        observed->exact_bytes != descriptor.exact_bytes ||
        !(observed->content_root == descriptor.content_root) ||
        !observed->regular_file || !observed->immutable ||
        observed->owner == nullptr) {
      return Status::FailedPrecondition(
          "runtime profile authority lease differs from canonical object");
    }
    owners.push_back(std::move(observed->owner));
  }
  return VerifiedRuntimeProfileAuthorityLeases(
      policy.policy_digest(), catalog.catalog_root(), envelope.envelope_root(),
      closure.closure_root(), graph.snapshot_root(), std::move(owners));
}

Result<RuntimeEngineAdmission> retain_runtime_engine_authority_leases(
    RuntimeEngineAdmission admission,
    VerifiedRuntimeProfileAuthorityLeases leases) {
  if (leases.size() != 5 ||
      !(leases.policy_digest() == admission.policy_digest_) ||
      !(leases.catalog_root() == admission.catalog_root_) ||
      !(leases.envelope_root() == admission.envelope_root_) ||
      !(leases.reference_closure_root() == admission.reference_closure_root_)) {
    return Status::FailedPrecondition(
        "runtime authority leases differ from engine admission");
  }
  admission.graph_snapshot_root_ = leases.graph_snapshot_root();
  admission.authority_anchor_ =
      std::make_shared<const VerifiedRuntimeProfileAuthorityLeases>(
          std::move(leases));
  return admission;
}

}  // namespace pih
