#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/deepseek_runtime_resolved_inputs.h"
#include "pih/model/runtime_profile_payload.h"

namespace pih {

using DeepSeekDsparkResolvedInputRoots = DeepSeekRuntimeResolvedInputRoots;

struct DeepSeekDsparkQualificationRoots final {
  Sha256Digest official_generation_receipt_root{};
  Sha256Digest official_checkpoint_numerical_receipt_root{};
  Sha256Digest three_stage_runtime_numerical_receipt_root{};
  Sha256Digest exact_profile_hardware_receipt_root{};
};

// This bounded public projection is produced by the release-evidence compiler.
// Its digest is the projection's domain_decision_roots_digest; it contains no
// private traces and deliberately excludes future profile/catalog roots.
class DeepSeekDsparkRuntimeEvidenceManifest final {
 public:
  static constexpr std::size_t kCanonicalBytes = 12 + 11 * 32;

  static Result<DeepSeekDsparkRuntimeEvidenceManifest> Create(
      RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
      RuntimeProfileResidency residency,
      DeepSeekDsparkResolvedInputRoots resolved_inputs,
      Sha256Digest artifact_root,
      DeepSeekDsparkQualificationRoots qualification_roots);

  [[nodiscard]] RuntimeProfileGpuFamily gpu_family() const noexcept {
    return gpu_family_;
  }
  [[nodiscard]] std::uint8_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] RuntimeProfileResidency residency() const noexcept {
    return residency_;
  }
  [[nodiscard]] const DeepSeekDsparkResolvedInputRoots& resolved_inputs()
      const noexcept { return resolved_inputs_; }
  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_root_;
  }
  [[nodiscard]] const DeepSeekDsparkQualificationRoots& qualification_roots()
      const noexcept { return qualification_roots_; }
  [[nodiscard]] std::span<const std::byte> canonical_bytes() const noexcept {
    return canonical_bytes_;
  }
  [[nodiscard]] const Sha256Digest& domain_decision_roots_digest()
      const noexcept { return domain_decision_roots_digest_; }

 private:
  DeepSeekDsparkRuntimeEvidenceManifest(
      RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
      RuntimeProfileResidency residency,
      DeepSeekDsparkResolvedInputRoots resolved_inputs,
      Sha256Digest artifact_root,
      DeepSeekDsparkQualificationRoots qualification_roots,
      std::vector<std::byte> canonical_bytes,
      Sha256Digest domain_decision_roots_digest) noexcept;

  RuntimeProfileGpuFamily gpu_family_ =
      RuntimeProfileGpuFamily::kH100Pcie80GiB;
  std::uint8_t world_size_ = 0;
  RuntimeProfileResidency residency_ = RuntimeProfileResidency::kHostSpill;
  DeepSeekDsparkResolvedInputRoots resolved_inputs_{};
  Sha256Digest artifact_root_{};
  DeepSeekDsparkQualificationRoots qualification_roots_{};
  std::vector<std::byte> canonical_bytes_;
  Sha256Digest domain_decision_roots_digest_{};
};

Result<DeepSeekDsparkRuntimeEvidenceManifest>
parse_deepseek_dspark_runtime_evidence_manifest(
    std::span<const std::byte> bytes);

class DeepSeekDsparkRuntimePermit final {
 public:
  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_root_;
  }
  [[nodiscard]] std::uint8_t world_size() const noexcept {
    return world_size_;
  }

 private:
  friend Result<DeepSeekDsparkRuntimePermit>
  issue_deepseek_dspark_runtime_permit(
      const RuntimeEngineAdmission&,
      const DeepSeekDsparkRuntimeEvidenceManifest&,
      const Sha256Digest&);
  friend Status validate_deepseek_dspark_runtime_permit(
      const DeepSeekDsparkRuntimePermit&, const RuntimeEngineAdmission&,
      const Sha256Digest&);

  DeepSeekDsparkRuntimePermit(
      Sha256Digest envelope_root, Sha256Digest reference_closure_root,
      Sha256Digest graph_snapshot_root,
      Sha256Digest device_observation_root, Sha256Digest artifact_root,
      Sha256Digest evidence_bundle_root,
      Sha256Digest domain_decision_roots_digest,
      RuntimeProfileResidency residency,
      std::vector<std::int32_t> device_ordinals) noexcept;

  Sha256Digest envelope_root_{};
  Sha256Digest reference_closure_root_{};
  Sha256Digest graph_snapshot_root_{};
  Sha256Digest device_observation_root_{};
  Sha256Digest artifact_root_{};
  Sha256Digest evidence_bundle_root_{};
  Sha256Digest domain_decision_roots_digest_{};
  std::uint8_t world_size_ = 0;
  RuntimeProfileResidency residency_ = RuntimeProfileResidency::kHostSpill;
  std::vector<std::int32_t> device_ordinals_;
};

Result<DeepSeekDsparkRuntimePermit> issue_deepseek_dspark_runtime_permit(
    const RuntimeEngineAdmission& admission,
    const DeepSeekDsparkRuntimeEvidenceManifest& evidence,
    const Sha256Digest& artifact_root);

Status validate_deepseek_dspark_runtime_permit(
    const DeepSeekDsparkRuntimePermit& permit,
    const RuntimeEngineAdmission& admission,
    const Sha256Digest& artifact_root);

}  // namespace pih
