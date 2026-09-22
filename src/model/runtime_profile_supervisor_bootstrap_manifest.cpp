#include "pih/model/runtime_profile_supervisor_bootstrap_manifest.h"

#include <string_view>

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) {
  std::byte aggregate{};
  for (const auto byte : value.bytes) aggregate |= byte;
  return aggregate != std::byte{};
}

void append_u64(std::vector<std::byte>& out, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::byte>((value >> shift) & 0xff));
  }
}

}  // namespace

RuntimeProfileSupervisorBootstrapManifest::
    RuntimeProfileSupervisorBootstrapManifest(
        RuntimeProfileInheritedFdManifest inherited_fds,
        std::uint64_t deployment_generation,
        Sha256Digest expected_envelope_root,
        Sha256Digest authority_snapshot_root,
        std::array<std::byte, kRuntimeReadinessNonceBytes> readiness_nonce,
        Sha256Digest readiness_nonce_digest,
        Sha256Digest manifest_root) noexcept
    : inherited_fds_(std::move(inherited_fds)),
      deployment_generation_(deployment_generation),
      expected_envelope_root_(expected_envelope_root),
      authority_snapshot_root_(authority_snapshot_root),
      readiness_nonce_(readiness_nonce),
      readiness_nonce_digest_(readiness_nonce_digest),
      manifest_root_(manifest_root) {}

Result<RuntimeProfileSupervisorBootstrapManifest>
RuntimeProfileSupervisorBootstrapManifest::Create(
    RuntimeProfileInheritedFdManifest inherited_fds,
    std::uint64_t deployment_generation,
    Sha256Digest expected_envelope_root,
    Sha256Digest authority_snapshot_root,
    std::array<std::byte, kRuntimeReadinessNonceBytes> readiness_nonce) {
  std::byte nonce_aggregate{};
  for (const auto byte : readiness_nonce) nonce_aggregate |= byte;
  if (deployment_generation == 0 || !nonzero(expected_envelope_root) ||
      !nonzero(authority_snapshot_root) || nonce_aggregate == std::byte{}) {
    return Status::InvalidArgument(
        "runtime supervisor bootstrap manifest identity is invalid");
  }
  auto nonce_digest = sha256(readiness_nonce);
  if (!nonce_digest.ok()) return nonce_digest.status();

  constexpr std::string_view abi =
      "runtime_profile_supervisor_bootstrap_manifest_v1";
  std::vector<std::byte> canonical;
  canonical.reserve(abi.size() + 8 + 32 * 4);
  const auto abi_bytes = std::as_bytes(std::span(abi.data(), abi.size()));
  canonical.insert(canonical.end(), abi_bytes.begin(), abi_bytes.end());
  append_u64(canonical, deployment_generation);
  const std::array<Sha256Digest, 4> bound_roots{
      expected_envelope_root, authority_snapshot_root,
      inherited_fds.manifest_root(), *nonce_digest};
  for (const auto& digest : bound_roots) {
    canonical.insert(canonical.end(), digest.bytes.begin(), digest.bytes.end());
  }
  auto root = sha256(canonical);
  if (!root.ok()) return root.status();
  return RuntimeProfileSupervisorBootstrapManifest(
      std::move(inherited_fds), deployment_generation, expected_envelope_root,
      authority_snapshot_root, readiness_nonce, *nonce_digest, *root);
}

bool RuntimeProfileSupervisorBootstrapManifest::matches_readiness_nonce(
    std::span<const std::byte> candidate) const noexcept {
  if (candidate.size() != readiness_nonce_.size()) return false;
  std::byte difference{};
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    difference |= candidate[index] ^ readiness_nonce_[index];
  }
  return difference == std::byte{};
}

}  // namespace pih
