#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_runtime_resolved_inputs.h"
#include "pih/model/runtime_profile_payload.h"

namespace pih {

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekDsparkRuntimeEvidenceManifest;
#endif

inline constexpr std::string_view kDeepSeekRuntimeArtifactEvidenceManifestAbi =
    "pih_deepseek_runtime_artifact_evidence_manifest_v1";
inline constexpr std::string_view kDeepSeekRuntimeArtifactAdmissionBindingAbi =
    "pih_deepseek_runtime_artifact_admission_binding_v1";

struct DeepSeekRuntimeArtifactQualificationRoots final {
  Sha256Digest official_generation_receipt_root{};
  Sha256Digest official_checkpoint_numerical_receipt_root{};
  Sha256Digest exact_profile_hardware_receipt_root{};
};

// Domain evidence for the canonical DeepSeek main path (D-Spark disabled).
// Its digest is committed as RuntimeEvidenceRoots::domain_decision_roots_digest.
class DeepSeekRuntimeArtifactEvidenceManifest final {
 public:
  static constexpr std::size_t kCanonicalBytes = 12 + 10 * 32;

  static Result<DeepSeekRuntimeArtifactEvidenceManifest> Create(
      RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
      RuntimeProfileResidency residency,
      DeepSeekRuntimeResolvedInputRoots resolved_inputs,
      Sha256Digest artifact_root,
      DeepSeekRuntimeArtifactQualificationRoots qualification_roots);

  [[nodiscard]] RuntimeProfileGpuFamily gpu_family() const noexcept {
    return gpu_family_;
  }
  [[nodiscard]] std::uint8_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] RuntimeProfileResidency residency() const noexcept {
    return residency_;
  }
  [[nodiscard]] const DeepSeekRuntimeResolvedInputRoots& resolved_inputs()
      const noexcept {
    return resolved_inputs_;
  }
  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_root_;
  }
  [[nodiscard]] const DeepSeekRuntimeArtifactQualificationRoots&
  qualification_roots() const noexcept {
    return qualification_roots_;
  }
  [[nodiscard]] std::span<const std::byte> canonical_bytes() const noexcept {
    return canonical_bytes_;
  }
  [[nodiscard]] const Sha256Digest& domain_decision_roots_digest()
      const noexcept {
    return domain_decision_roots_digest_;
  }

 private:
  DeepSeekRuntimeArtifactEvidenceManifest(
      RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
      RuntimeProfileResidency residency,
      DeepSeekRuntimeResolvedInputRoots resolved_inputs,
      Sha256Digest artifact_root,
      DeepSeekRuntimeArtifactQualificationRoots qualification_roots,
      std::vector<std::byte> canonical_bytes,
      Sha256Digest domain_decision_roots_digest) noexcept;

  RuntimeProfileGpuFamily gpu_family_ =
      RuntimeProfileGpuFamily::kRtx4090D24GiB;
  std::uint8_t world_size_ = 0;
  RuntimeProfileResidency residency_ = RuntimeProfileResidency::kHostSpill;
  DeepSeekRuntimeResolvedInputRoots resolved_inputs_{};
  Sha256Digest artifact_root_{};
  DeepSeekRuntimeArtifactQualificationRoots qualification_roots_{};
  std::vector<std::byte> canonical_bytes_;
  Sha256Digest domain_decision_roots_digest_{};
};

Result<DeepSeekRuntimeArtifactEvidenceManifest>
parse_deepseek_runtime_artifact_evidence_manifest(
    std::span<const std::byte> bytes);

// Common admission-retaining antecedent for the later transfer owner. It does
// not prove that descriptors were sent, adopted, mapped, or made model-ready.
class DeepSeekRuntimeArtifactAdmissionBinding final {
 public:
  static Result<DeepSeekRuntimeArtifactAdmissionBinding> Issue(
      const RuntimeEngineAdmission& admission,
      const DeepSeekRuntimeArtifactEvidenceManifest& evidence);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  static Result<DeepSeekRuntimeArtifactAdmissionBinding> Issue(
      const RuntimeEngineAdmission& admission,
      const DeepSeekDsparkRuntimeEvidenceManifest& evidence);
#endif

  DeepSeekRuntimeArtifactAdmissionBinding(
      const DeepSeekRuntimeArtifactAdmissionBinding&) = delete;
  DeepSeekRuntimeArtifactAdmissionBinding& operator=(
      const DeepSeekRuntimeArtifactAdmissionBinding&) = delete;
  DeepSeekRuntimeArtifactAdmissionBinding(
      DeepSeekRuntimeArtifactAdmissionBinding&&) noexcept = default;
  DeepSeekRuntimeArtifactAdmissionBinding& operator=(
      DeepSeekRuntimeArtifactAdmissionBinding&&) noexcept = default;

  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_root_;
  }
  [[nodiscard]] const Sha256Digest& evidence_manifest_root() const noexcept {
    return evidence_manifest_root_;
  }
  [[nodiscard]] const Sha256Digest& binding_root() const noexcept {
    return binding_root_;
  }
  [[nodiscard]] bool admission_authority_retained() const noexcept {
    return admission_anchor_ != nullptr && admission_anchor_->production_ready();
  }
  [[nodiscard]] Status validate(
      const RuntimeEngineAdmission& admission,
      const Sha256Digest& artifact_root) const;

 private:
  DeepSeekRuntimeArtifactAdmissionBinding(
      Sha256Digest artifact_root, Sha256Digest evidence_manifest_root,
      Sha256Digest binding_root,
      std::shared_ptr<const RuntimeEngineAdmission> admission_anchor) noexcept;

  Sha256Digest artifact_root_{};
  Sha256Digest evidence_manifest_root_{};
  Sha256Digest binding_root_{};
  std::shared_ptr<const RuntimeEngineAdmission> admission_anchor_;
};

}  // namespace pih
