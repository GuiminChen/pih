#include "pih/model/qwen3_hardware_evidence.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <utility>

namespace pih {
namespace {

Result<std::size_t> gate_index(QwenHardwareGate gate) {
  const auto value = static_cast<std::uint8_t>(gate);
  if (value > static_cast<std::uint8_t>(QwenHardwareGate::kBoundedMemory)) {
    return Status::InvalidArgument("unknown Qwen hardware evidence gate");
  }
  return static_cast<std::size_t>(value);
}

}  // namespace

Result<QwenHardwareEvidence> QwenHardwareEvidence::Create(
    QwenNumericalRunIdentity identity) {
  return QwenHardwareEvidence(std::move(identity));
}

Status QwenHardwareEvidence::record(
    const QwenNumericalRunIdentity& identity, QwenHardwareGate gate,
    QwenHardwareGateResult gate_result) {
  if (!(identity == identity_)) {
    return Status::FailedPrecondition(
        "Qwen hardware evidence identity cannot be spliced");
  }
  auto index = gate_index(gate);
  if (!index.ok()) return index.status();
  if (state_ != QwenHardwareEvidenceState::kHardwareEvidenceOpen) {
    return Status::FailedPrecondition("Qwen hardware evidence is terminal");
  }
  if (recorded_[index.value()]) {
    return Status::FailedPrecondition(
        "Qwen hardware evidence gate was already recorded");
  }
  const bool digest_nonzero = std::any_of(
      gate_result.raw_artifact_digest.bytes.begin(),
      gate_result.raw_artifact_digest.bytes.end(),
      [](std::byte value) { return value != std::byte{0}; });
  if (!digest_nonzero || gate_result.raw_artifact_bytes == 0) {
    return Status::InvalidArgument("Qwen hardware gate artifact is invalid");
  }
  artifact_digests_[index.value()] = gate_result.raw_artifact_digest;
  artifact_bytes_[index.value()] = gate_result.raw_artifact_bytes;
  if (!gate_result.passed) {
    state_ = QwenHardwareEvidenceState::kCandidateFailed;
    return Status::FailedPrecondition("Qwen hardware evidence gate failed");
  }
  recorded_[index.value()] = true;
  if (std::all_of(recorded_.begin(), recorded_.end(),
                  [](bool value) { return value; })) {
    state_ = QwenHardwareEvidenceState::kHardwareGatesComplete;
  }
  return Status::Ok();
}

std::uint32_t QwenHardwareEvidence::completed_gate_count() const noexcept {
  return static_cast<std::uint32_t>(
      std::count(recorded_.begin(), recorded_.end(), true));
}

Result<Sha256Digest> QwenHardwareEvidence::evidence_bundle_digest() const {
  if (!eligible_for_external_attestation()) {
    return Status::FailedPrecondition(
        "Qwen hardware evidence bundle is incomplete");
  }
  auto identity_digest = identity_.semantic_digest();
  if (!identity_digest.ok()) return identity_digest.status();
  std::string canonical = "pih.qwen_hardware_evidence_bundle.v1\n" +
                          identity_digest->hex();
  for (std::size_t i = 0; i < artifact_digests_.size(); ++i) {
    canonical += "\n" + std::to_string(i) + ":" + artifact_digests_[i].hex() +
                 ":" + std::to_string(artifact_bytes_[i]);
  }
  return sha256(std::as_bytes(std::span(canonical.data(), canonical.size())));
}

}  // namespace pih
