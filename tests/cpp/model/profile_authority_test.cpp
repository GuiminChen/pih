#include "pih/model/profile_authority.h"

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace pih {
namespace {

std::vector<std::byte> bytes(std::string_view value) {
  const auto raw = std::as_bytes(std::span(value.data(), value.size()));
  return {raw.begin(), raw.end()};
}

Sha256Digest digest(std::string_view value) {
  return sha256(bytes(value)).value();
}

ProfileSignature signature(std::string key = "release-key-1") {
  return {std::move(key), std::vector<std::byte>(64, std::byte{7})};
}

std::array<std::byte, 32> public_key() {
  std::array<std::byte, 32> value{};
  value.fill(std::byte{3});
  return value;
}

ProfileTrustKey trust_key(
    ProfileSignerRole role = ProfileSignerRole::kRelease,
    bool revoked = false) {
  return {"release-key-1", role, revoked, public_key()};
}

SignedProfileEnvelope envelope(std::uint64_t generation = 7) {
  return SignedProfileEnvelope::Create(
             "qwen-exact-1", "r1", generation, bytes("payload"),
             digest("closure"), {signature()})
      .value();
}

SignedProfileCatalog catalog(const SignedProfileEnvelope& value,
                             std::uint64_t generation = 7) {
  return SignedProfileCatalog::Create(
             generation, {{"qwen-exact-1", "r1", value.envelope_root(),
               value.envelope_bytes()}},
             {signature()})
      .value();
}

ProfileTrustPolicy policy(const SignedProfileCatalog& value) {
  return ProfileTrustPolicy::Create(
             7, value.catalog_root(), {trust_key()})
      .value();
}

class Verifier final : public ProfileEd25519Verifier {
 public:
  Status verify(std::string_view key_id,
                std::span<const std::byte> public_key_bytes,
                std::span<const std::byte> message,
                std::span<const std::byte> signature_bytes) override {
    ++calls;
    keys.emplace_back(key_id);
    public_key_sizes.push_back(public_key_bytes.size());
    message_sizes.push_back(message.size());
    signature_sizes.push_back(signature_bytes.size());
    return result;
  }

  Status result = Status::Ok();
  int calls = 0;
  std::vector<std::string> keys;
  std::vector<std::size_t> public_key_sizes;
  std::vector<std::size_t> message_sizes;
  std::vector<std::size_t> signature_sizes;
};

TEST(ProfileAuthorityTest, ExactSignedCatalogJoinProducesVerifiedReceipt) {
  auto profile = envelope();
  auto index = catalog(profile);
  auto trust = policy(index);
  Verifier verifier;

  auto receipt = verify_profile_authority(
      trust, index, profile, "qwen-exact-1", "r1", verifier);

  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_TRUE(receipt->verified());
  EXPECT_EQ(receipt->profile_id(), "qwen-exact-1");
  EXPECT_EQ(receipt->revision(), "r1");
  EXPECT_EQ(receipt->catalog_generation(), 7U);
  EXPECT_EQ(receipt->catalog_root(), index.catalog_root());
  EXPECT_EQ(receipt->envelope_root(), profile.envelope_root());
  EXPECT_EQ(receipt->payload_root(), profile.payload_root());
  EXPECT_EQ(std::vector<std::byte>(profile.payload_bytes().begin(),
                                   profile.payload_bytes().end()),
            bytes("payload"));
  EXPECT_EQ(receipt->reference_closure_root(), digest("closure"));
  EXPECT_EQ(verifier.calls, 2);
  EXPECT_EQ(verifier.message_sizes,
            (std::vector<std::size_t>{
                index.canonical_signed_bytes().size(),
                profile.canonical_signed_bytes().size()}));
  EXPECT_EQ(verifier.public_key_sizes,
            (std::vector<std::size_t>{32, 32}));
  EXPECT_EQ(verifier.signature_sizes,
            (std::vector<std::size_t>{64, 64}));
}

TEST(ProfileAuthorityTest, RejectsGenerationRollbackAndCatalogSplice) {
  auto profile = envelope();
  auto index = catalog(profile);
  Verifier verifier;

  auto minimum_too_new = ProfileTrustPolicy::Create(
      8, std::nullopt, {trust_key()}).value();
  EXPECT_FALSE(verify_profile_authority(
      minimum_too_new, index, profile, "qwen-exact-1", "r1", verifier).ok());
  EXPECT_EQ(verifier.calls, 0);

  auto wrong_root = ProfileTrustPolicy::Create(
      7, digest("other-catalog"), {trust_key()}).value();
  EXPECT_FALSE(verify_profile_authority(
      wrong_root, index, profile, "qwen-exact-1", "r1", verifier).ok());
  EXPECT_EQ(verifier.calls, 0);

  auto other_profile = envelope(8);
  auto cross_generation = catalog(other_profile, 8);
  auto trust = ProfileTrustPolicy::Create(
      7, cross_generation.catalog_root(), {trust_key()}).value();
  EXPECT_FALSE(verify_profile_authority(
      trust, cross_generation, profile, "qwen-exact-1", "r1", verifier).ok());
  EXPECT_EQ(verifier.calls, 1);

  auto spliced = SignedProfileCatalog::Create(
      7, {{"qwen-exact-1", "r1", digest("wrong-envelope"),
        profile.envelope_bytes()}},
      {signature()}).value();
  trust = ProfileTrustPolicy::Create(
      7, spliced.catalog_root(), {trust_key()}).value();
  verifier.calls = 0;
  EXPECT_FALSE(verify_profile_authority(
      trust, spliced, profile, "qwen-exact-1", "r1", verifier).ok());
  EXPECT_EQ(verifier.calls, 1);
}

TEST(ProfileAuthorityTest, RejectsRevokedWrongRoleAndBadSignatures) {
  auto profile = envelope();
  auto index = catalog(profile);
  Verifier verifier;

  for (const auto key : std::array{
           trust_key(ProfileSignerRole::kRelease, true),
           trust_key(ProfileSignerRole::kBuilder, false)}) {
    auto trust = ProfileTrustPolicy::Create(
        7, index.catalog_root(), {key}).value();
    EXPECT_FALSE(verify_profile_authority(
        trust, index, profile, "qwen-exact-1", "r1", verifier).ok());
  }
  EXPECT_EQ(verifier.calls, 0);

  verifier.result = Status::FailedPrecondition("bad Ed25519 signature");
  auto trust = policy(index);
  EXPECT_FALSE(verify_profile_authority(
      trust, index, profile, "qwen-exact-1", "r1", verifier).ok());
  EXPECT_EQ(verifier.calls, 1);
}

TEST(ProfileAuthorityTest, EnforcesClosedParserObjectBounds) {
  EXPECT_FALSE(ProfileTrustPolicy::Create(
      1, std::nullopt, {}).ok());
  std::vector<ProfileTrustKey> too_many_keys;
  for (int index = 0; index < 33; ++index) {
    too_many_keys.push_back(
        {"key-" + std::to_string(index), ProfileSignerRole::kRelease, false,
         public_key()});
  }
  EXPECT_FALSE(ProfileTrustPolicy::Create(
      1, std::nullopt, std::move(too_many_keys)).ok());

  std::vector<ProfileSignature> too_many_signatures(9, signature());
  EXPECT_FALSE(SignedProfileEnvelope::Create(
      "qwen-exact-1", "r1", 1, bytes("payload"), digest("closure"),
      std::move(too_many_signatures)).ok());

  auto profile = envelope();
  EXPECT_FALSE(SignedProfileCatalog::Create(
      7, {{"qwen-exact-1", "r1", profile.envelope_root(),
        profile.envelope_bytes()},
       {"qwen-exact-1", "r1", profile.envelope_root(),
        profile.envelope_bytes()}},
      {signature()}).ok());

  auto duplicate_signatures =
      std::vector<ProfileSignature>{signature(), signature()};
  auto duplicate_envelope = SignedProfileEnvelope::Create(
      "qwen-exact-1", "r1", 7, bytes("payload"), digest("closure"),
      std::move(duplicate_signatures));
  ASSERT_TRUE(duplicate_envelope.ok());
  auto duplicate_catalog = catalog(*duplicate_envelope);
  Verifier verifier;
  auto receipt = verify_profile_authority(
      policy(duplicate_catalog), duplicate_catalog, *duplicate_envelope,
      "qwen-exact-1", "r1", verifier);
  ASSERT_TRUE(receipt.ok());
  EXPECT_EQ(verifier.calls, 2);

  auto different = signature();
  different.bytes[0] = std::byte{9};
  EXPECT_FALSE(SignedProfileEnvelope::Create(
      "qwen-exact-1", "r1", 7, bytes("payload"), digest("closure"),
      {signature(), std::move(different)}).ok());
}

}  // namespace
}  // namespace pih
