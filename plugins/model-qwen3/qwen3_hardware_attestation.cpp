#include "pih/model/qwen3_hardware_attestation.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace pih {
namespace {

bool nonzero(const Sha256Digest& digest) {
  return std::any_of(digest.bytes.begin(), digest.bytes.end(),
                     [](std::byte value) { return value != std::byte{0}; });
}

}  // namespace

Result<QwenHardwareAttestationReceipt> QwenHardwareAttestationReceipt::Create(
    QwenNumericalRunIdentity identity, Sha256Digest evidence_bundle_digest,
    const QwenTargetObservation& target_observation,
    std::string signer_key_id,
    std::span<const std::byte> signature) {
  if (!nonzero(evidence_bundle_digest) ||
      !(identity == target_observation.identity()) || signer_key_id.empty() ||
      signer_key_id.size() > 128 || signature.empty() ||
      signature.size() > 4096) {
    return Status::InvalidArgument("Qwen hardware attestation receipt is invalid");
  }
  auto identity_digest = identity.semantic_digest();
  if (!identity_digest.ok()) return identity_digest.status();
  const std::string payload =
      "pih.qwen_bf16_hardware_attestation.v1\n" +
      identity_digest->hex() + "\n" + evidence_bundle_digest.hex() + "\n" +
      target_observation.semantic_digest().hex() + "\n" + signer_key_id;
  auto payload_digest =
      sha256(std::as_bytes(std::span(payload.data(), payload.size())));
  if (!payload_digest.ok()) return payload_digest.status();
  return QwenHardwareAttestationReceipt(
      std::move(identity), evidence_bundle_digest, payload_digest.value(),
      std::vector<std::byte>(signature.begin(), signature.end()));
}

Result<QwenSupportedHardware> attest_qwen_hardware(
    const QwenHardwareEvidence& evidence,
    const QwenHardwareAttestationReceipt& receipt,
    QwenAttestationVerifier& verifier) {
  if (!evidence.eligible_for_external_attestation() ||
      !evidence.matches_identity(receipt.identity())) {
    return Status::FailedPrecondition(
        "Qwen hardware evidence is not attestable for this receipt");
  }
  auto bundle_digest = evidence.evidence_bundle_digest();
  if (!bundle_digest.ok()) return bundle_digest.status();
  if (!(bundle_digest.value() == receipt.evidence_bundle_digest())) {
    return Status::FailedPrecondition(
        "Qwen attestation evidence bundle root mismatched");
  }
  const Status verified =
      verifier.verify(receipt.payload_digest(), receipt.signature());
  if (!verified.ok()) return verified;
  return QwenSupportedHardware(receipt.identity(), receipt.payload_digest());
}

}  // namespace pih
