#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pih/model/runtime_profile_reference_closure.h"
#include "pih/model/sealed_reachable_object_dag.h"

namespace pih {

enum class RuntimeProfileAuthorityRole : std::uint8_t {
  kTrustPolicy = 1,
  kCatalog = 2,
  kEnvelope = 3,
  kReferenceClosure = 4,
  kDigestDagManifest = 5,
};

struct RuntimeProfileAuthorityDescriptor final {
  RuntimeProfileAuthorityRole role = RuntimeProfileAuthorityRole::kTrustPolicy;
  std::uint64_t exact_bytes = 0;
  Sha256Digest content_root{};
};

class RuntimeProfileAuthorityLeaseOwner {
 public:
  virtual ~RuntimeProfileAuthorityLeaseOwner() = default;
};

struct RuntimeProfileAuthorityLeaseObservation final {
  RuntimeProfileAuthorityRole role = RuntimeProfileAuthorityRole::kTrustPolicy;
  std::uint64_t exact_bytes = 0;
  Sha256Digest content_root{};
  bool regular_file = false;
  bool immutable = false;
  std::shared_ptr<const RuntimeProfileAuthorityLeaseOwner> owner;
};

class RuntimeProfileAuthorityLeaseProbe {
 public:
  virtual ~RuntimeProfileAuthorityLeaseProbe() = default;
  virtual Result<RuntimeProfileAuthorityLeaseObservation> observe(
      const RuntimeProfileAuthorityDescriptor& descriptor) = 0;
};

class VerifiedRuntimeProfileAuthorityLeases final {
 public:
  [[nodiscard]] std::size_t size() const noexcept { return owners_.size(); }
  [[nodiscard]] const Sha256Digest& policy_digest() const noexcept {
    return policy_digest_;
  }
  [[nodiscard]] const Sha256Digest& catalog_root() const noexcept {
    return catalog_root_;
  }
  [[nodiscard]] const Sha256Digest& envelope_root() const noexcept {
    return envelope_root_;
  }
  [[nodiscard]] const Sha256Digest& reference_closure_root() const noexcept {
    return reference_closure_root_;
  }
  [[nodiscard]] const Sha256Digest& graph_snapshot_root() const noexcept {
    return graph_snapshot_root_;
  }

 private:
  friend Result<VerifiedRuntimeProfileAuthorityLeases>
  verify_runtime_profile_authority_leases(
      const ProfileTrustPolicy&, const SignedProfileCatalog&,
      const SignedProfileEnvelope&, const RuntimeProfileReferenceClosure&,
      const SealedReachableObjectDag&,
      RuntimeProfileAuthorityLeaseProbe&);
  VerifiedRuntimeProfileAuthorityLeases(
      Sha256Digest policy_digest, Sha256Digest catalog_root,
      Sha256Digest envelope_root, Sha256Digest reference_closure_root,
      Sha256Digest graph_snapshot_root,
      std::vector<std::shared_ptr<const RuntimeProfileAuthorityLeaseOwner>>
          owners) noexcept;
  Sha256Digest policy_digest_{};
  Sha256Digest catalog_root_{};
  Sha256Digest envelope_root_{};
  Sha256Digest reference_closure_root_{};
  Sha256Digest graph_snapshot_root_{};
  std::vector<std::shared_ptr<const RuntimeProfileAuthorityLeaseOwner>> owners_;
};

Result<VerifiedRuntimeProfileAuthorityLeases>
verify_runtime_profile_authority_leases(
    const ProfileTrustPolicy& policy,
    const SignedProfileCatalog& catalog,
    const SignedProfileEnvelope& envelope,
    const RuntimeProfileReferenceClosure& closure,
    const SealedReachableObjectDag& graph,
    RuntimeProfileAuthorityLeaseProbe& probe);

}  // namespace pih
