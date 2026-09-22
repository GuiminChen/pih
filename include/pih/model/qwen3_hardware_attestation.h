#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pih/model/qwen3_hardware_evidence.h"
#include "pih/model/qwen3_target_observation.h"

namespace pih {

class QwenAttestationVerifier {
 public:
  virtual ~QwenAttestationVerifier() = default;
  virtual Status verify(const Sha256Digest& payload_digest,
                        std::span<const std::byte> signature) = 0;
};

class QwenHardwareAttestationReceipt final {
 public:
  static Result<QwenHardwareAttestationReceipt> Create(
      QwenNumericalRunIdentity identity, Sha256Digest evidence_bundle_digest,
      const QwenTargetObservation& target_observation,
      std::string signer_key_id,
      std::span<const std::byte> signature);

  [[nodiscard]] const QwenNumericalRunIdentity& identity() const noexcept {
    return identity_;
  }
  [[nodiscard]] const Sha256Digest& payload_digest() const noexcept {
    return payload_digest_;
  }
  [[nodiscard]] const Sha256Digest& evidence_bundle_digest() const noexcept {
    return evidence_bundle_digest_;
  }
  [[nodiscard]] std::span<const std::byte> signature() const noexcept {
    return signature_;
  }

 private:
  QwenHardwareAttestationReceipt(QwenNumericalRunIdentity identity,
                                 Sha256Digest evidence_bundle_digest,
                                 Sha256Digest payload_digest,
                                 std::vector<std::byte> signature)
      : identity_(std::move(identity)),
        evidence_bundle_digest_(evidence_bundle_digest),
        payload_digest_(payload_digest),
        signature_(std::move(signature)) {}

  QwenNumericalRunIdentity identity_;
  Sha256Digest evidence_bundle_digest_;
  Sha256Digest payload_digest_;
  std::vector<std::byte> signature_;
};

class QwenSupportedHardware final {
 public:
  [[nodiscard]] bool supported() const noexcept { return true; }
  [[nodiscard]] const QwenNumericalRunIdentity& identity() const noexcept {
    return identity_;
  }
  [[nodiscard]] const Sha256Digest& attestation_payload_digest() const noexcept {
    return attestation_payload_digest_;
  }

 private:
  friend Result<QwenSupportedHardware> attest_qwen_hardware(
      const QwenHardwareEvidence&, const QwenHardwareAttestationReceipt&,
      QwenAttestationVerifier&);
  QwenSupportedHardware(QwenNumericalRunIdentity identity,
                        Sha256Digest attestation_payload_digest)
      : identity_(std::move(identity)),
        attestation_payload_digest_(attestation_payload_digest) {}

  QwenNumericalRunIdentity identity_;
  Sha256Digest attestation_payload_digest_;
};

Result<QwenSupportedHardware> attest_qwen_hardware(
    const QwenHardwareEvidence& evidence,
    const QwenHardwareAttestationReceipt& receipt,
    QwenAttestationVerifier& verifier);

}  // namespace pih
