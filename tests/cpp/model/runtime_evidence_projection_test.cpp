#include "pih/model/runtime_evidence_projection.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest evidence_root(std::uint8_t value) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(value));
  return result;
}

RuntimeEvidenceRoots evidence_roots() {
  return {evidence_root(1), evidence_root(2), evidence_root(3),
          evidence_root(4), evidence_root(5)};
}

ProfileSignature evidence_signature() {
  return {"release-1",
          std::vector<std::byte>(kEd25519SignatureBytes, std::byte{0x5a})};
}

class EvidenceVerifier final : public ProfileEd25519Verifier {
 public:
  Status verify(std::string_view, std::span<const std::byte>,
                std::span<const std::byte>,
                std::span<const std::byte>) override { return Status::Ok(); }
};

VerifiedRuntimeProfile evidence_profile(const Sha256Digest& projection_root) {
  RuntimeProfileRoots profile_roots{
      evidence_root(11), evidence_root(12), evidence_root(13), projection_root,
      evidence_root(15), evidence_root(16), evidence_root(17)};
  auto payload = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative,
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 2, true,
      RuntimeProfileResidency::kHostSpill,
      RuntimeProfileEvidenceState::kCorrectnessSupported,
      profile_roots).value();
  auto envelope = SignedProfileEnvelope::Create(
      "deepseek", "r1", 9, payload.canonical_bytes(), evidence_root(20),
      {evidence_signature()}).value();
  auto catalog = SignedProfileCatalog::Create(
      9, {{"deepseek", "r1", envelope.envelope_root(),
           envelope.envelope_bytes()}}, {evidence_signature()}).value();
  ProfileTrustKey key{};
  key.key_id = "release-1";
  key.role = ProfileSignerRole::kRelease;
  key.public_key.fill(std::byte{0x22});
  auto policy = ProfileTrustPolicy::Create(
      9, catalog.catalog_root(), {key}).value();
  EvidenceVerifier verifier;
  auto authority = verify_profile_authority(
      policy, catalog, envelope, "deepseek", "r1", verifier).value();
  return bind_verified_runtime_profile(authority, envelope).value();
}

TEST(RuntimeEvidenceProjectionTest, RoundTripsOnlyClosedStatesAndRoots) {
  auto value = RuntimeEvidenceProjection::Create(
      RuntimeEvidenceBundleState::kComplete,
      RuntimeEvidenceReplayState::kPassed,
      RuntimeHardwareQualificationState::kSupported, evidence_roots());
  ASSERT_TRUE(value.ok());
  auto parsed = parse_runtime_evidence_projection(value->canonical_bytes());
  ASSERT_TRUE(parsed.ok());
  EXPECT_EQ(parsed->projection_root(), value->projection_root());
  auto trailing = std::vector<std::byte>(value->canonical_bytes().begin(),
                                         value->canonical_bytes().end());
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(parse_runtime_evidence_projection(trailing).ok());
  auto roots = evidence_roots();
  roots.audit_closure_root = {};
  EXPECT_FALSE(RuntimeEvidenceProjection::Create(
                   RuntimeEvidenceBundleState::kComplete,
                   RuntimeEvidenceReplayState::kPassed,
                   RuntimeHardwareQualificationState::kSupported, roots).ok());
}

TEST(RuntimeEvidenceProjectionTest, SupportedProfileRequiresCompletePassedHardware) {
  auto supported = RuntimeEvidenceProjection::Create(
      RuntimeEvidenceBundleState::kComplete,
      RuntimeEvidenceReplayState::kPassed,
      RuntimeHardwareQualificationState::kSupported, evidence_roots()).value();
  auto profile = evidence_profile(supported.projection_root());
  EXPECT_TRUE(verify_runtime_evidence_projection(profile, supported).ok());

  auto uncalibrated = RuntimeEvidenceProjection::Create(
      RuntimeEvidenceBundleState::kComplete,
      RuntimeEvidenceReplayState::kPassed,
      RuntimeHardwareQualificationState::kUncalibrated,
      evidence_roots()).value();
  auto uncalibrated_profile = evidence_profile(uncalibrated.projection_root());
  EXPECT_FALSE(verify_runtime_evidence_projection(
                   uncalibrated_profile, uncalibrated).ok());
  auto failed = RuntimeEvidenceProjection::Create(
      RuntimeEvidenceBundleState::kComplete,
      RuntimeEvidenceReplayState::kFailed,
      RuntimeHardwareQualificationState::kSupported, evidence_roots()).value();
  auto failed_profile = evidence_profile(failed.projection_root());
  EXPECT_FALSE(verify_runtime_evidence_projection(failed_profile, failed).ok());
}

}  // namespace
}  // namespace pih
