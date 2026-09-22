#include "pih/model/qwen3_hardware_evidence.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

std::string run_json(std::uint64_t generation = 7) {
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

QwenNumericalRunIdentity identity(std::uint64_t generation = 7) {
  return QwenNumericalRunIdentity::Parse(run_json(generation)).value();
}

constexpr QwenHardwareGate kGates[] = {
    QwenHardwareGate::kArtifactIdentity,
    QwenHardwareGate::kAllKernelFamilies,
    QwenHardwareGate::kLayerTaps,
    QwenHardwareGate::kFullLogits,
    QwenHardwareGate::kGreedyTrajectory,
    QwenHardwareGate::kKvState,
    QwenHardwareGate::kComputeSanitizer,
    QwenHardwareGate::kBoundedMemory,
};

QwenHardwareGateResult gate_result(std::uint8_t value, bool passed = true) {
  return {passed, Sha256Digest::ParseHex(std::string(64, "12345678"[value % 8]))
                      .value(),
          static_cast<std::uint64_t>(value) + 1};
}

TEST(QwenHardwareEvidenceTest, MissingGateRemainsHardwareOpen) {
  auto evidence = QwenHardwareEvidence::Create(identity()).value();
  for (std::size_t i = 0; i + 1 < std::size(kGates); ++i) {
    ASSERT_TRUE(evidence.record(identity(), kGates[i], gate_result(i)).ok());
  }
  EXPECT_EQ(evidence.state(), QwenHardwareEvidenceState::kHardwareEvidenceOpen);
  EXPECT_FALSE(evidence.eligible_for_external_attestation());
}

TEST(QwenHardwareEvidenceTest, ExactCompleteRunAwaitsExternalAttestation) {
  auto evidence = QwenHardwareEvidence::Create(identity()).value();
  for (std::uint8_t i = 0; i < std::size(kGates); ++i) {
    ASSERT_TRUE(evidence.record(identity(), kGates[i], gate_result(i)).ok());
  }
  EXPECT_EQ(evidence.completed_gate_count(), 8);
  EXPECT_EQ(evidence.state(),
            QwenHardwareEvidenceState::kHardwareGatesComplete);
  EXPECT_TRUE(evidence.eligible_for_external_attestation());
  EXPECT_TRUE(evidence.evidence_bundle_digest().ok());
  EXPECT_FALSE(evidence.record(identity(), kGates[0], gate_result(0)).ok());
}

TEST(QwenHardwareEvidenceTest, FailureIsTerminalAndCannotBeRepaired) {
  auto evidence = QwenHardwareEvidence::Create(identity()).value();
  EXPECT_FALSE(
      evidence
          .record(identity(), QwenHardwareGate::kFullLogits,
                  gate_result(3, false))
          .ok());
  EXPECT_EQ(evidence.state(), QwenHardwareEvidenceState::kCandidateFailed);
  EXPECT_FALSE(
      evidence
          .record(identity(), QwenHardwareGate::kFullLogits, gate_result(3))
          .ok());
}

TEST(QwenHardwareEvidenceTest, RejectsCrossRunAndUnknownGate) {
  auto evidence = QwenHardwareEvidence::Create(identity()).value();
  EXPECT_FALSE(evidence
                   .record(identity(8), QwenHardwareGate::kArtifactIdentity,
                           gate_result(0))
                   .ok());
  EXPECT_FALSE(evidence
                   .record(identity(), static_cast<QwenHardwareGate>(255),
                           gate_result(0))
                   .ok());
  EXPECT_EQ(evidence.completed_gate_count(), 0);
  EXPECT_FALSE(evidence.evidence_bundle_digest().ok());
}

}  // namespace
}  // namespace pih
