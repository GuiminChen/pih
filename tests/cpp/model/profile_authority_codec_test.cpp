#include "pih/model/profile_authority_codec.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

namespace pih {
namespace {

Sha256Digest codec_digest(std::byte seed) {
  Sha256Digest value{};
  value.bytes.fill(seed);
  return value;
}

ProfileSignature codec_signature() {
  return {"release-1", std::vector<std::byte>(kEd25519SignatureBytes,
                                               std::byte{0x5a})};
}

TEST(ProfileAuthorityCodecTest, ParsesOnlyExactCanonicalObjects) {
  ProfileTrustKey key{};
  key.key_id = "release-1";
  key.role = ProfileSignerRole::kRelease;
  key.public_key.fill(std::byte{0x33});
  auto policy = ProfileTrustPolicy::Create(11, codec_digest(std::byte{0x44}),
                                            {key});
  ASSERT_TRUE(policy.ok());
  auto parsed_policy =
      parse_profile_trust_policy(policy->canonical_object_bytes());
  ASSERT_TRUE(parsed_policy.ok());
  EXPECT_EQ(parsed_policy->policy_digest(), policy->policy_digest());

  const std::array payload{std::byte{1}, std::byte{2}, std::byte{3}};
  auto envelope = SignedProfileEnvelope::Create(
      "qwen3-0.6b", "r1", 11, payload, codec_digest(std::byte{0x66}),
      {codec_signature()});
  ASSERT_TRUE(envelope.ok());
  EXPECT_EQ(envelope->envelope_bytes(),
            envelope->canonical_object_bytes().size());
  auto parsed_envelope =
      parse_signed_profile_envelope(envelope->canonical_object_bytes());
  ASSERT_TRUE(parsed_envelope.ok());
  EXPECT_EQ(parsed_envelope->envelope_root(), envelope->envelope_root());
  EXPECT_EQ(parsed_envelope->payload_root(), envelope->payload_root());
  EXPECT_TRUE(std::equal(parsed_envelope->payload_bytes().begin(),
                         parsed_envelope->payload_bytes().end(),
                         payload.begin(), payload.end()));

  auto catalog = SignedProfileCatalog::Create(
      11, {{"qwen3-0.6b", "r1", envelope->envelope_root(),
            envelope->envelope_bytes()}},
      {codec_signature()});
  ASSERT_TRUE(catalog.ok());
  auto parsed_catalog =
      parse_signed_profile_catalog(catalog->canonical_object_bytes());
  ASSERT_TRUE(parsed_catalog.ok());
  EXPECT_EQ(parsed_catalog->catalog_root(), catalog->catalog_root());
  EXPECT_FALSE(parsed_catalog->graph_bound());

  auto graph_catalog = SignedProfileCatalog::CreateGraphBound(
      11, {codec_digest(std::byte{0x77}), 3, 2},
      {{"qwen3-0.6b", "r1", envelope->envelope_root(),
        envelope->envelope_bytes()}},
      {codec_signature()});
  ASSERT_TRUE(graph_catalog.ok());
  auto parsed_graph_catalog = parse_signed_profile_catalog(
      graph_catalog->canonical_object_bytes());
  ASSERT_TRUE(parsed_graph_catalog.ok());
  ASSERT_TRUE(parsed_graph_catalog->graph_bound());
  EXPECT_EQ(parsed_graph_catalog->catalog_root(), graph_catalog->catalog_root());
  EXPECT_EQ(parsed_graph_catalog->graph_binding()->snapshot_root,
            codec_digest(std::byte{0x77}));
  EXPECT_EQ(parsed_graph_catalog->graph_binding()->node_count, 3U);
  EXPECT_EQ(parsed_graph_catalog->graph_binding()->edge_count, 2U);

  auto rebound_graph_catalog = SignedProfileCatalog::CreateGraphBound(
      11, {codec_digest(std::byte{0x78}), 3, 2},
      {{"qwen3-0.6b", "r1", envelope->envelope_root(),
        envelope->envelope_bytes()}},
      {codec_signature()});
  ASSERT_TRUE(rebound_graph_catalog.ok());
  EXPECT_FALSE(std::equal(
      rebound_graph_catalog->canonical_signed_bytes().begin(),
      rebound_graph_catalog->canonical_signed_bytes().end(),
      graph_catalog->canonical_signed_bytes().begin(),
      graph_catalog->canonical_signed_bytes().end()));
  EXPECT_NE(rebound_graph_catalog->catalog_root(), graph_catalog->catalog_root());
}

TEST(ProfileAuthorityCodecTest, RejectsTruncationTrailingAndNoncanonicalOrder) {
  const std::array payload{std::byte{1}};
  auto envelope = SignedProfileEnvelope::Create(
      "qwen3-0.6b", "r1", 11, payload, codec_digest(std::byte{0x66}),
      {codec_signature()});
  ASSERT_TRUE(envelope.ok());
  auto bytes = std::vector<std::byte>(envelope->canonical_object_bytes().begin(),
                                      envelope->canonical_object_bytes().end());

  EXPECT_FALSE(parse_signed_profile_envelope(
                   std::span<const std::byte>(bytes).first(bytes.size() - 1))
                   .ok());
  bytes.push_back(std::byte{0});
  EXPECT_FALSE(parse_signed_profile_envelope(bytes).ok());

  auto first = codec_signature();
  first.key_id = "z-key";
  auto second = codec_signature();
  second.key_id = "a-key";
  auto canonical = SignedProfileEnvelope::Create(
      "qwen3-0.6b", "r1", 11, payload, codec_digest(std::byte{0x66}),
      {first, second});
  ASSERT_TRUE(canonical.ok());
  auto noncanonical = std::vector<std::byte>(canonical->canonical_object_bytes().begin(),
                                             canonical->canonical_object_bytes().end());
  const auto a = std::search(noncanonical.begin(), noncanonical.end(),
                             std::as_bytes(std::span("a-key", 5)).begin(),
                             std::as_bytes(std::span("a-key", 5)).end());
  const auto z = std::search(noncanonical.begin(), noncanonical.end(),
                             std::as_bytes(std::span("z-key", 5)).begin(),
                             std::as_bytes(std::span("z-key", 5)).end());
  ASSERT_NE(a, noncanonical.end());
  ASSERT_NE(z, noncanonical.end());
  std::swap_ranges(a, a + 5, z);
  EXPECT_FALSE(parse_signed_profile_envelope(noncanonical).ok());
}

TEST(ProfileAuthorityCodecTest, RejectsDeclaredLengthsBeforeAllocation) {
  const auto domain =
      std::as_bytes(std::span("pih:profile-envelope-object:v1", 36));
  std::vector<std::byte> malformed(domain.begin(), domain.end());
  malformed.push_back(std::byte{0});
  malformed.insert(malformed.end(), 8, std::byte{0xff});
  EXPECT_FALSE(parse_signed_profile_envelope(malformed).ok());
}

}  // namespace
}  // namespace pih
