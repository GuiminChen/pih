#include "pih/model/runtime_profile_reference_lease.h"

namespace pih {

VerifiedRuntimeProfileReferenceLeases::VerifiedRuntimeProfileReferenceLeases(
    Sha256Digest closure_root, RuntimeProfileRoots reference_roots,
    std::uint64_t capacity_template_exact_bytes,
    std::vector<std::shared_ptr<const RuntimeProfileReferenceLeaseOwner>>
        owners) noexcept
    : closure_root_(closure_root), reference_roots_(reference_roots),
      capacity_template_exact_bytes_(capacity_template_exact_bytes),
      owners_(std::move(owners)) {}

Result<VerifiedRuntimeProfileReferenceLeases>
verify_runtime_profile_reference_leases(
    const RuntimeProfileReferenceClosure& closure,
    RuntimeProfileReferenceLeaseProbe& probe) {
  std::vector<std::shared_ptr<const RuntimeProfileReferenceLeaseOwner>> owners;
  owners.reserve(closure.descriptors().size());
  RuntimeProfileRoots roots{};
  std::uint64_t capacity_template_exact_bytes = 0;
  for (const auto& descriptor : closure.descriptors()) {
    auto observation = probe.observe(descriptor);
    if (!observation.ok()) return observation.status();
    if (observation->role != descriptor.role ||
        observation->exact_bytes != descriptor.exact_bytes ||
        !(observation->content_root == descriptor.object_root) ||
        !observation->regular_file || !observation->immutable ||
        observation->owner == nullptr) {
      return Status::FailedPrecondition(
          "runtime reference lease differs from signed descriptor");
    }
    owners.push_back(std::move(observation->owner));
    switch (descriptor.role) {
      case RuntimeProfileReferenceRole::kRuntimeSemantic:
        roots.runtime_semantic_root = descriptor.object_root; break;
      case RuntimeProfileReferenceRole::kCapacityTemplate:
        roots.capacity_template_root = descriptor.object_root;
        capacity_template_exact_bytes = descriptor.exact_bytes;
        break;
      case RuntimeProfileReferenceRole::kFeatureSelection:
        roots.feature_selection_root = descriptor.object_root; break;
      case RuntimeProfileReferenceRole::kReleaseEvidence:
        roots.release_evidence_root = descriptor.object_root; break;
      case RuntimeProfileReferenceRole::kHardwareIdentity:
        roots.hardware_identity_root = descriptor.object_root; break;
      case RuntimeProfileReferenceRole::kKernelClosure:
        roots.kernel_closure_root = descriptor.object_root; break;
      case RuntimeProfileReferenceRole::kSecurityRuntime:
        roots.security_runtime_root = descriptor.object_root; break;
    }
  }
  if (capacity_template_exact_bytes == 0) {
    return Status::FailedPrecondition(
        "runtime capacity template descriptor is absent");
  }
  return VerifiedRuntimeProfileReferenceLeases(
      closure.closure_root(), roots, capacity_template_exact_bytes,
      std::move(owners));
}

Result<RuntimeEngineAdmission> retain_runtime_engine_reference_leases(
    RuntimeEngineAdmission admission,
    VerifiedRuntimeProfileReferenceLeases leases) {
  if (leases.size() != 7) {
    return Status::FailedPrecondition(
        "runtime engine requires seven verified reference leases");
  }
  if (!(leases.closure_root() == admission.reference_closure_root_) ||
      !(leases.reference_roots().runtime_semantic_root ==
            admission.reference_roots_.runtime_semantic_root) ||
      !(leases.reference_roots().capacity_template_root ==
            admission.reference_roots_.capacity_template_root) ||
      !(leases.reference_roots().feature_selection_root ==
            admission.reference_roots_.feature_selection_root) ||
      !(leases.reference_roots().release_evidence_root ==
            admission.reference_roots_.release_evidence_root) ||
      !(leases.reference_roots().hardware_identity_root ==
            admission.reference_roots_.hardware_identity_root) ||
      !(leases.reference_roots().kernel_closure_root ==
            admission.reference_roots_.kernel_closure_root) ||
      !(leases.reference_roots().security_runtime_root ==
            admission.reference_roots_.security_runtime_root)) {
    return Status::FailedPrecondition(
        "runtime reference leases differ from engine admission");
  }
  admission.reference_retention_ =
      RuntimeProfileReferenceRetention::kImmutableDescriptorLease;
  admission.capacity_template_exact_bytes_ =
      leases.capacity_template_exact_bytes();
  admission.reference_anchor_ =
      std::make_shared<const VerifiedRuntimeProfileReferenceLeases>(
          std::move(leases));
  return admission;
}

}  // namespace pih
