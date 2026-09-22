#pragma once

#include <span>
#include <vector>

#include "pih/model/runtime_profile_payload.h"

namespace pih {

enum class RuntimeEvidenceBundleState : std::uint8_t {
  kAssembling = 1,
  kComplete = 2,
  kInvalid = 3,
  kRevoked = 4,
};

enum class RuntimeEvidenceReplayState : std::uint8_t {
  kNotRun = 1,
  kPassed = 2,
  kFailed = 3,
};

enum class RuntimeHardwareQualificationState : std::uint8_t {
  kSupported = 1,
  kUncalibrated = 2,
  kRejected = 3,
};

class RuntimeEvidenceProjection final {
 public:
  static Result<RuntimeEvidenceProjection> Create(
      RuntimeEvidenceBundleState bundle_state,
      RuntimeEvidenceReplayState replay_state,
      RuntimeHardwareQualificationState hardware_state,
      RuntimeEvidenceRoots roots);
  [[nodiscard]] RuntimeEvidenceBundleState bundle_state() const noexcept {
    return bundle_state_;
  }
  [[nodiscard]] RuntimeEvidenceReplayState replay_state() const noexcept {
    return replay_state_;
  }
  [[nodiscard]] RuntimeHardwareQualificationState hardware_state() const noexcept {
    return hardware_state_;
  }
  [[nodiscard]] const RuntimeEvidenceRoots& roots() const noexcept {
    return roots_;
  }
  [[nodiscard]] std::span<const std::byte> canonical_bytes() const noexcept {
    return canonical_bytes_;
  }
  [[nodiscard]] const Sha256Digest& projection_root() const noexcept {
    return projection_root_;
  }

 private:
  RuntimeEvidenceProjection(RuntimeEvidenceBundleState bundle_state,
                            RuntimeEvidenceReplayState replay_state,
                            RuntimeHardwareQualificationState hardware_state,
                            RuntimeEvidenceRoots roots,
                            std::vector<std::byte> canonical_bytes,
                            Sha256Digest projection_root) noexcept;
  RuntimeEvidenceBundleState bundle_state_ =
      RuntimeEvidenceBundleState::kAssembling;
  RuntimeEvidenceReplayState replay_state_ =
      RuntimeEvidenceReplayState::kNotRun;
  RuntimeHardwareQualificationState hardware_state_ =
      RuntimeHardwareQualificationState::kUncalibrated;
  RuntimeEvidenceRoots roots_{};
  std::vector<std::byte> canonical_bytes_;
  Sha256Digest projection_root_{};
};

Result<RuntimeEvidenceProjection> parse_runtime_evidence_projection(
    std::span<const std::byte> bytes);
Status verify_runtime_evidence_projection(
    const VerifiedRuntimeProfile& profile,
    const RuntimeEvidenceProjection& projection);
Result<RuntimeEngineAdmission> bind_runtime_engine_evidence_projection(
    RuntimeEngineAdmission admission,
    const RuntimeEvidenceProjection& projection);

}  // namespace pih
