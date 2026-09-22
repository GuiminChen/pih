#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include "pih/model/qwen3_numerical_run.h"

namespace pih {

enum class QwenHardwareGate : std::uint8_t {
  kArtifactIdentity = 0,
  kAllKernelFamilies,
  kLayerTaps,
  kFullLogits,
  kGreedyTrajectory,
  kKvState,
  kComputeSanitizer,
  kBoundedMemory,
};

enum class QwenHardwareEvidenceState : std::uint8_t {
  kHardwareEvidenceOpen = 0,
  kCandidateFailed,
  kHardwareGatesComplete,
};

struct QwenHardwareGateResult final {
  bool passed;
  Sha256Digest raw_artifact_digest;
  std::uint64_t raw_artifact_bytes;
};

class QwenHardwareEvidence final {
 public:
  static Result<QwenHardwareEvidence> Create(QwenNumericalRunIdentity identity);

  Status record(const QwenNumericalRunIdentity& identity,
                QwenHardwareGate gate, QwenHardwareGateResult result);
  [[nodiscard]] QwenHardwareEvidenceState state() const noexcept {
    return state_;
  }
  [[nodiscard]] bool eligible_for_external_attestation() const noexcept {
    return state_ == QwenHardwareEvidenceState::kHardwareGatesComplete;
  }
  [[nodiscard]] std::uint32_t completed_gate_count() const noexcept;
  Result<Sha256Digest> evidence_bundle_digest() const;
  [[nodiscard]] bool matches_identity(
      const QwenNumericalRunIdentity& identity) const noexcept {
    return identity_ == identity;
  }

 private:
  explicit QwenHardwareEvidence(QwenNumericalRunIdentity identity)
      : identity_(std::move(identity)) {}

  QwenNumericalRunIdentity identity_;
  std::array<bool, 8> recorded_{};
  std::array<Sha256Digest, 8> artifact_digests_{};
  std::array<std::uint64_t, 8> artifact_bytes_{};
  QwenHardwareEvidenceState state_ =
      QwenHardwareEvidenceState::kHardwareEvidenceOpen;
};

}  // namespace pih
