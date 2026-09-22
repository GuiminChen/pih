#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/runtime_profile_inherited_fd_manifest.h"

namespace pih {

inline constexpr std::size_t kRuntimeReadinessNonceBytes = 32;

class RuntimeProfileSupervisorBootstrapManifest final {
 public:
  static Result<RuntimeProfileSupervisorBootstrapManifest> Create(
      RuntimeProfileInheritedFdManifest inherited_fds,
      std::uint64_t deployment_generation,
      Sha256Digest expected_envelope_root,
      Sha256Digest authority_snapshot_root,
      std::array<std::byte, kRuntimeReadinessNonceBytes> readiness_nonce);

  [[nodiscard]] const RuntimeProfileInheritedFdManifest& inherited_fds()
      const noexcept { return inherited_fds_; }
  [[nodiscard]] std::uint64_t deployment_generation() const noexcept {
    return deployment_generation_;
  }
  [[nodiscard]] const Sha256Digest& expected_envelope_root() const noexcept {
    return expected_envelope_root_;
  }
  [[nodiscard]] const Sha256Digest& authority_snapshot_root() const noexcept {
    return authority_snapshot_root_;
  }
  [[nodiscard]] const Sha256Digest& readiness_nonce_digest() const noexcept {
    return readiness_nonce_digest_;
  }
  [[nodiscard]] const Sha256Digest& manifest_root() const noexcept {
    return manifest_root_;
  }
  [[nodiscard]] bool matches_readiness_nonce(
      std::span<const std::byte> candidate) const noexcept;

 private:
  RuntimeProfileSupervisorBootstrapManifest(
      RuntimeProfileInheritedFdManifest inherited_fds,
      std::uint64_t deployment_generation,
      Sha256Digest expected_envelope_root,
      Sha256Digest authority_snapshot_root,
      std::array<std::byte, kRuntimeReadinessNonceBytes> readiness_nonce,
      Sha256Digest readiness_nonce_digest,
      Sha256Digest manifest_root) noexcept;

  RuntimeProfileInheritedFdManifest inherited_fds_;
  std::uint64_t deployment_generation_ = 0;
  Sha256Digest expected_envelope_root_{};
  Sha256Digest authority_snapshot_root_{};
  std::array<std::byte, kRuntimeReadinessNonceBytes> readiness_nonce_{};
  Sha256Digest readiness_nonce_digest_{};
  Sha256Digest manifest_root_{};
};

}  // namespace pih
