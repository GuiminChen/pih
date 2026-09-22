#pragma once

#include <span>

#include "pih/model/runtime_profile_payload.h"
#include "pih/model/runtime_profile_supervisor_bootstrap_manifest.h"

namespace pih {

class RuntimeProfileReadinessReceipt final {
 public:
  [[nodiscard]] std::uint64_t deployment_generation() const noexcept {
    return deployment_generation_;
  }
  [[nodiscard]] const Sha256Digest& authority_snapshot_root() const noexcept {
    return authority_snapshot_root_;
  }
  [[nodiscard]] const Sha256Digest& bootstrap_manifest_root() const noexcept {
    return bootstrap_manifest_root_;
  }
  [[nodiscard]] const Sha256Digest& device_observation_root() const noexcept {
    return device_observation_root_;
  }
  [[nodiscard]] const Sha256Digest& capacity_plan_instance_root() const noexcept {
    return capacity_plan_instance_root_;
  }
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
  [[nodiscard]] const Sha256Digest& readiness_nonce_digest() const noexcept {
    return readiness_nonce_digest_;
  }
  [[nodiscard]] const Sha256Digest& receipt_root() const noexcept {
    return receipt_root_;
  }

 private:
  friend Result<RuntimeProfileReadinessReceipt>
  issue_runtime_profile_readiness_receipt(
      const RuntimeEngineAdmission&,
      const RuntimeProfileSupervisorBootstrapManifest&,
      std::span<const std::byte>, const Sha256Digest&, const Sha256Digest&);
  RuntimeProfileReadinessReceipt(
      std::uint64_t deployment_generation,
      Sha256Digest authority_snapshot_root,
      Sha256Digest bootstrap_manifest_root,
      Sha256Digest device_observation_root,
      Sha256Digest capacity_plan_instance_root,
      Sha256Digest policy_digest, Sha256Digest catalog_root,
      Sha256Digest envelope_root, Sha256Digest reference_closure_root,
      Sha256Digest graph_snapshot_root,
      Sha256Digest readiness_nonce_digest,
      Sha256Digest receipt_root) noexcept;

  std::uint64_t deployment_generation_ = 0;
  Sha256Digest authority_snapshot_root_{};
  Sha256Digest bootstrap_manifest_root_{};
  Sha256Digest device_observation_root_{};
  Sha256Digest capacity_plan_instance_root_{};
  Sha256Digest policy_digest_{};
  Sha256Digest catalog_root_{};
  Sha256Digest envelope_root_{};
  Sha256Digest reference_closure_root_{};
  Sha256Digest graph_snapshot_root_{};
  Sha256Digest readiness_nonce_digest_{};
  Sha256Digest receipt_root_{};
};

Result<RuntimeProfileReadinessReceipt> issue_runtime_profile_readiness_receipt(
    const RuntimeEngineAdmission& admission,
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
    std::span<const std::byte> readiness_nonce,
    const Sha256Digest& active_authority_snapshot_root,
    const Sha256Digest& capacity_plan_instance_root);

}  // namespace pih
