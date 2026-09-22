#include "pih/model/runtime_profile_readiness_receipt.h"

#include <string_view>
#include <vector>

namespace pih {
namespace {

bool readiness_nonzero(const Sha256Digest& value) {
  std::byte aggregate{};
  for (const auto byte : value.bytes) aggregate |= byte;
  return aggregate != std::byte{};
}

void readiness_append_u64(std::vector<std::byte>& out, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::byte>((value >> shift) & 0xff));
  }
}

}  // namespace

RuntimeProfileReadinessReceipt::RuntimeProfileReadinessReceipt(
    std::uint64_t deployment_generation,
    Sha256Digest authority_snapshot_root,
    Sha256Digest bootstrap_manifest_root,
    Sha256Digest device_observation_root,
    Sha256Digest capacity_plan_instance_root,
    Sha256Digest policy_digest, Sha256Digest catalog_root,
    Sha256Digest envelope_root, Sha256Digest reference_closure_root,
    Sha256Digest graph_snapshot_root,
    Sha256Digest readiness_nonce_digest,
    Sha256Digest receipt_root) noexcept
    : deployment_generation_(deployment_generation),
      authority_snapshot_root_(authority_snapshot_root),
      bootstrap_manifest_root_(bootstrap_manifest_root),
      device_observation_root_(device_observation_root),
      capacity_plan_instance_root_(capacity_plan_instance_root),
      policy_digest_(policy_digest), catalog_root_(catalog_root),
      envelope_root_(envelope_root),
      reference_closure_root_(reference_closure_root),
      graph_snapshot_root_(graph_snapshot_root),
      readiness_nonce_digest_(readiness_nonce_digest),
      receipt_root_(receipt_root) {}

Result<RuntimeProfileReadinessReceipt> issue_runtime_profile_readiness_receipt(
    const RuntimeEngineAdmission& admission,
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
    std::span<const std::byte> readiness_nonce,
    const Sha256Digest& active_authority_snapshot_root,
    const Sha256Digest& capacity_plan_instance_root) {
  if (!admission.production_compute_ready()) {
    return Status::FailedPrecondition(
        "runtime admission is not production compute ready");
  }
  if (!(admission.envelope_root() == bootstrap.expected_envelope_root()) ||
      !(active_authority_snapshot_root == bootstrap.authority_snapshot_root())) {
    return Status::FailedPrecondition(
        "runtime readiness authority changed during startup");
  }
  if (!bootstrap.matches_readiness_nonce(readiness_nonce)) {
    return Status::FailedPrecondition("runtime readiness nonce does not match");
  }
  if (!readiness_nonzero(capacity_plan_instance_root)) {
    return Status::InvalidArgument(
        "runtime capacity plan instance root is empty");
  }

  constexpr std::string_view abi = "runtime_profile_readiness_receipt_v2";
  std::vector<std::byte> canonical;
  canonical.reserve(abi.size() + 8 + 32 * 10);
  const auto abi_bytes = std::as_bytes(std::span(abi.data(), abi.size()));
  canonical.insert(canonical.end(), abi_bytes.begin(), abi_bytes.end());
  readiness_append_u64(canonical, bootstrap.deployment_generation());
  for (const auto& root :
       {bootstrap.manifest_root(), admission.envelope_root(),
        admission.device_observation_root(), active_authority_snapshot_root,
        capacity_plan_instance_root, bootstrap.readiness_nonce_digest(),
        admission.policy_digest(), admission.catalog_root(),
        admission.reference_closure_root(), admission.graph_snapshot_root()}) {
    canonical.insert(canonical.end(), root.bytes.begin(), root.bytes.end());
  }
  auto receipt_root = sha256(canonical);
  if (!receipt_root.ok()) return receipt_root.status();
  return RuntimeProfileReadinessReceipt(
      bootstrap.deployment_generation(), active_authority_snapshot_root,
      bootstrap.manifest_root(), admission.device_observation_root(),
      capacity_plan_instance_root, admission.policy_digest(),
      admission.catalog_root(), admission.envelope_root(),
      admission.reference_closure_root(), admission.graph_snapshot_root(),
      bootstrap.readiness_nonce_digest(),
      *receipt_root);
}

}  // namespace pih
