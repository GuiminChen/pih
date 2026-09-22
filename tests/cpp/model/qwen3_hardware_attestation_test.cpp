#include "pih/model/qwen3_hardware_attestation.h"
#include "pih/model/qwen3_target_observation.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

std::string attestation_run_json(std::uint64_t generation = 7) {
  return "{\"schema\":\"pih.qwen_bf16_numerical_run.v1\","
         "\"model_sha256\":\"" + std::string(64, 'a') +
         "\",\"fixture_sha256\":\"" + std::string(64, 'b') +
         "\",\"tolerance_sha256\":\"" + std::string(64, 'c') +
         "\",\"kernel_bundle_sha256\":\"" + std::string(64, 'd') +
         "\",\"build_sha256\":\"" + std::string(64, 'e') +
         "\",\"environment_sha256\":\"" + std::string(64, 'f') +
         "\",\"target_gpu\":\"RTX_4090_D\",\"target_sm\":\"sm_89\","
         "\"run_generation\":" + std::to_string(generation) + "}";
}

QwenNumericalRunIdentity run_identity(std::uint64_t generation = 7) {
  return QwenNumericalRunIdentity::Parse(attestation_run_json(generation))
      .value();
}

QwenHardwareEvidence complete_evidence() {
  auto evidence = QwenHardwareEvidence::Create(run_identity()).value();
  for (std::uint8_t i = 0; i < 8; ++i) {
    const auto artifact =
        Sha256Digest::ParseHex(std::string(64, "12345678"[i])).value();
    EXPECT_TRUE(evidence
                    .record(run_identity(), static_cast<QwenHardwareGate>(i),
                            {true, artifact,
                             static_cast<std::uint64_t>(i) + 1})
                    .ok());
  }
  return evidence;
}

Sha256Digest digest(char value) {
  return Sha256Digest::ParseHex(std::string(64, value)).value();
}

QwenTargetObservation target_observation(
    QwenNumericalRunIdentity identity = run_identity()) {
  QwenObservedDevice device{};
  device.visible_device_count = 1;
  device.current_ordinal = 0;
  device.name = "NVIDIA GeForce RTX 4090 D";
  device.compute_major = 8;
  device.compute_minor = 9;
  device.total_global_memory_bytes = 24ULL * 1024 * 1024 * 1024;
  device.uuid[0] = std::byte{1};
  device.pci_bus = 3;
  device.driver_version = 13020;
  device.runtime_version = 13020;
  return QwenTargetObservation::Create(std::move(identity), std::move(device))
      .value();
}

class FakeAttestationVerifier final : public QwenAttestationVerifier {
 public:
  Status verify(const Sha256Digest& payload_digest,
                std::span<const std::byte> signature) override {
    ++calls;
    seen_payload = payload_digest;
    seen_signature_bytes = signature.size();
    return result;
  }
  Status result = Status::Ok();
  int calls = 0;
  Sha256Digest seen_payload{};
  std::size_t seen_signature_bytes = 0;
};

TEST(QwenHardwareAttestationTest, ValidExternalReceiptProducesSupportedRecord) {
  auto evidence = complete_evidence();
  const std::byte signature[] = {std::byte{1}, std::byte{2}};
  auto receipt = QwenHardwareAttestationReceipt::Create(
      run_identity(), evidence.evidence_bundle_digest().value(),
      target_observation(), "release-key-1",
      signature);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  FakeAttestationVerifier verifier;
  auto supported = attest_qwen_hardware(evidence, receipt.value(), verifier);
  ASSERT_TRUE(supported.ok()) << supported.status().message();
  EXPECT_TRUE(supported->supported());
  EXPECT_EQ(supported->identity().target_gpu(), QwenTargetGpu::kRtx4090D);
  EXPECT_EQ(supported->identity().target_sm(), 89);
  EXPECT_EQ(supported->attestation_payload_digest(), receipt->payload_digest());
  EXPECT_EQ(verifier.calls, 1);
  EXPECT_EQ(verifier.seen_signature_bytes, 2);
}

TEST(QwenHardwareAttestationTest, RejectsIncompleteCrossRunAndBadSignature) {
  auto incomplete = QwenHardwareEvidence::Create(run_identity()).value();
  const std::byte signature[] = {std::byte{1}};
  auto receipt = QwenHardwareAttestationReceipt::Create(
                     run_identity(), digest('1'), target_observation(),
                     "release-key-1",
                     signature)
                     .value();
  FakeAttestationVerifier verifier;
  EXPECT_FALSE(attest_qwen_hardware(incomplete, receipt, verifier).ok());
  EXPECT_EQ(verifier.calls, 0);

  auto cross_run = QwenHardwareAttestationReceipt::Create(
                       run_identity(8), digest('1'),
                       target_observation(run_identity(8)),
                       "release-key-1", signature)
                       .value();
  auto complete = complete_evidence();
  EXPECT_FALSE(attest_qwen_hardware(complete, receipt, verifier).ok());
  EXPECT_EQ(verifier.calls, 0);

  receipt = QwenHardwareAttestationReceipt::Create(
                run_identity(), complete.evidence_bundle_digest().value(),
                target_observation(), "release-key-1", signature)
                .value();
  EXPECT_FALSE(attest_qwen_hardware(complete, cross_run, verifier).ok());
  EXPECT_EQ(verifier.calls, 0);

  verifier.result = Status::FailedPrecondition("bad signature");
  EXPECT_FALSE(attest_qwen_hardware(complete, receipt, verifier).ok());
  EXPECT_EQ(verifier.calls, 1);
}

TEST(QwenHardwareAttestationTest, RejectsEmptyAuthorityAndSignature) {
  const std::byte signature[] = {std::byte{1}};
  EXPECT_FALSE(QwenHardwareAttestationReceipt::Create(
                   run_identity(), digest('1'), target_observation(), {},
                   signature)
                   .ok());
  EXPECT_FALSE(QwenHardwareAttestationReceipt::Create(
                   run_identity(), digest('1'), target_observation(), "key", {})
                   .ok());
  EXPECT_FALSE(QwenHardwareAttestationReceipt::Create(
                   run_identity(8), digest('1'), target_observation(), "key",
                   signature)
                   .ok());
}

}  // namespace
}  // namespace pih
