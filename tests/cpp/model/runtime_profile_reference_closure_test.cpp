#include "pih/model/runtime_profile_reference_closure.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace pih {
namespace {

Sha256Digest root(std::uint8_t value) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(value));
  return result;
}

RuntimeProfileRoots roots() {
  return {root(1), root(2), root(3), root(4), root(5), root(6), root(7)};
}

std::vector<RuntimeProfileReferenceDescriptor> descriptors() {
  std::vector<RuntimeProfileReferenceDescriptor> result;
  for (std::uint8_t role = 1; role <= 7; ++role) {
    result.push_back({static_cast<RuntimeProfileReferenceRole>(role),
                      "schema_v1", static_cast<std::uint64_t>(100 + role),
                      root(role)});
  }
  return result;
}

ProfileSignature signature() {
  return {"release-1",
          std::vector<std::byte>(kEd25519SignatureBytes, std::byte{0x5a})};
}

class AcceptingVerifier final : public ProfileEd25519Verifier {
 public:
  Status verify(std::string_view, std::span<const std::byte>,
                std::span<const std::byte>,
                std::span<const std::byte>) override {
    return Status::Ok();
  }
};

TEST(RuntimeProfileReferenceClosureTest, RoundTripsSevenCanonicalRoles) {
  auto value = RuntimeProfileReferenceClosure::Create(descriptors());
  ASSERT_TRUE(value.ok());
  auto parsed = parse_runtime_profile_reference_closure(value->canonical_bytes());
  ASSERT_TRUE(parsed.ok());
  EXPECT_EQ(parsed->descriptors().size(), 7U);
  EXPECT_EQ(parsed->closure_root(), value->closure_root());

  auto trailing = std::vector<std::byte>(value->canonical_bytes().begin(),
                                         value->canonical_bytes().end());
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(parse_runtime_profile_reference_closure(trailing).ok());
  auto duplicate = descriptors();
  duplicate.back().role = RuntimeProfileReferenceRole::kKernelClosure;
  EXPECT_FALSE(RuntimeProfileReferenceClosure::Create(std::move(duplicate)).ok());
  auto reused = descriptors();
  reused.back().object_root = reused.front().object_root;
  EXPECT_FALSE(RuntimeProfileReferenceClosure::Create(std::move(reused)).ok());
}

TEST(RuntimeProfileReferenceClosureTest, BindsEnvelopeAndPayloadRoots) {
  auto closure = RuntimeProfileReferenceClosure::Create(descriptors()).value();
  auto payload = RuntimeProfilePayload::Create(
      RuntimeProfileModel::kDeepSeekV4Flash0731,
      RuntimeProfileWeightFormat::kDeepSeekNative,
      RuntimeProfileGpuFamily::kH100Pcie80GiB, 2, true,
      RuntimeProfileResidency::kHostSpill,
      RuntimeProfileEvidenceState::kCorrectnessSupported, roots()).value();
  auto envelope = SignedProfileEnvelope::Create(
      "deepseek", "r1", 9, payload.canonical_bytes(), closure.closure_root(),
      {signature()}).value();
  auto catalog = SignedProfileCatalog::Create(
      9, {{"deepseek", "r1", envelope.envelope_root(),
           envelope.envelope_bytes()}}, {signature()}).value();
  ProfileTrustKey key{};
  key.key_id = "release-1";
  key.role = ProfileSignerRole::kRelease;
  key.public_key.fill(std::byte{0x22});
  auto policy = ProfileTrustPolicy::Create(
      9, catalog.catalog_root(), {key}).value();
  AcceptingVerifier verifier;
  auto authority = verify_profile_authority(
      policy, catalog, envelope, "deepseek", "r1", verifier).value();
  auto profile = bind_verified_runtime_profile(authority, envelope).value();
  EXPECT_TRUE(verify_runtime_profile_reference_closure(
                  profile, envelope, closure).ok());

  auto wrong = descriptors();
  wrong[3].object_root = root(8);
  auto wrong_closure = RuntimeProfileReferenceClosure::Create(std::move(wrong));
  ASSERT_TRUE(wrong_closure.ok());
  EXPECT_FALSE(verify_runtime_profile_reference_closure(
                   profile, envelope, *wrong_closure).ok());
}

TEST(RuntimeProfileReferenceClosureTest, VerifiesEveryReferencedObject) {
  std::vector<std::vector<std::byte>> objects;
  std::vector<RuntimeProfileReferenceDescriptor> records;
  for (std::uint8_t role = 1; role <= 7; ++role) {
    objects.emplace_back(role, static_cast<std::byte>(role));
    records.push_back({static_cast<RuntimeProfileReferenceRole>(role),
                       "schema_v1", role, sha256(objects.back()).value()});
  }
  auto closure = RuntimeProfileReferenceClosure::Create(std::move(records));
  ASSERT_TRUE(closure.ok());
  EXPECT_TRUE(verify_runtime_profile_reference_objects(*closure, objects).ok());
  objects[2].push_back(std::byte{3});
  EXPECT_FALSE(verify_runtime_profile_reference_objects(*closure, objects).ok());
  objects[2].pop_back();
  objects[2][0] = std::byte{0};
  EXPECT_FALSE(verify_runtime_profile_reference_objects(*closure, objects).ok());
  EXPECT_FALSE(verify_runtime_profile_reference_objects(
                   *closure,
                   std::span<const std::vector<std::byte>>(objects).first(6))
                   .ok());
}

}  // namespace
}  // namespace pih
