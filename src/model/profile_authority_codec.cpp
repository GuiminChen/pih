#include "pih/model/profile_authority_codec.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace pih {
namespace {

class Reader final {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  Status domain(std::string_view expected) {
    if (remaining() < expected.size() + 1) return invalid();
    const auto expected_bytes = std::as_bytes(std::span(expected));
    if (!std::equal(expected_bytes.begin(), expected_bytes.end(),
                    bytes_.begin() + static_cast<std::ptrdiff_t>(offset_)) ||
        bytes_[offset_ + expected.size()] != std::byte{0}) {
      return invalid();
    }
    offset_ += expected.size() + 1;
    return Status::Ok();
  }

  Result<std::uint8_t> u8() {
    if (remaining() < 1) return invalid();
    return std::to_integer<std::uint8_t>(bytes_[offset_++]);
  }

  Result<std::uint32_t> u32() {
    if (remaining() < 4) return invalid();
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
      value |= static_cast<std::uint32_t>(
                   std::to_integer<std::uint8_t>(bytes_[offset_++]))
               << (index * 8U);
    }
    return value;
  }

  Result<std::uint64_t> u64() {
    if (remaining() < 8) return invalid();
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
      value |= static_cast<std::uint64_t>(
                   std::to_integer<std::uint8_t>(bytes_[offset_++]))
               << (index * 8U);
    }
    return value;
  }

  Result<std::span<const std::byte>> fixed(std::size_t count) {
    if (count > remaining()) return invalid();
    auto value = bytes_.subspan(offset_, count);
    offset_ += count;
    return value;
  }

  Result<std::span<const std::byte>> blob(std::size_t maximum) {
    auto count = u64();
    if (!count.ok()) return count.status();
    if (*count > maximum || *count > remaining() ||
        *count > std::numeric_limits<std::size_t>::max()) {
      return invalid();
    }
    return fixed(static_cast<std::size_t>(*count));
  }

  Result<std::string> text() {
    auto value = blob(128);
    if (!value.ok()) return value.status();
    return std::string(reinterpret_cast<const char*>(value->data()),
                       value->size());
  }

  [[nodiscard]] bool done() const noexcept { return offset_ == bytes_.size(); }

 private:
  [[nodiscard]] std::size_t remaining() const noexcept {
    return bytes_.size() - offset_;
  }
  static Status invalid() {
    return Status::InvalidArgument("profile authority object encoding is invalid");
  }

  std::span<const std::byte> bytes_;
  std::size_t offset_ = 0;
};

Status invalid_encoding() {
  return Status::InvalidArgument("profile authority object encoding is invalid");
}

Result<std::vector<ProfileSignature>> parse_signatures(Reader& reader) {
  auto count = reader.u32();
  if (!count.ok()) return count.status();
  if (*count == 0 || *count > kProfileMaximumSignatures) {
    return invalid_encoding();
  }
  std::vector<ProfileSignature> signatures;
  signatures.reserve(*count);
  for (std::uint32_t index = 0; index < *count; ++index) {
    auto key_id = reader.text();
    if (!key_id.ok()) return key_id.status();
    auto bytes = reader.blob(kEd25519SignatureBytes);
    if (!bytes.ok() || bytes->size() != kEd25519SignatureBytes) {
      return invalid_encoding();
    }
    signatures.push_back(
        {std::move(*key_id), std::vector<std::byte>(bytes->begin(), bytes->end())});
  }
  return signatures;
}

struct SealedObject final {
  std::span<const std::byte> signed_bytes;
  std::vector<ProfileSignature> signatures;
};

Result<SealedObject> parse_sealed(std::span<const std::byte> bytes,
                                  std::string_view domain,
                                  std::size_t signed_maximum) {
  Reader reader(bytes);
  auto status = reader.domain(domain);
  if (!status.ok()) return status;
  auto signed_bytes = reader.blob(signed_maximum);
  if (!signed_bytes.ok()) return signed_bytes.status();
  auto signatures = parse_signatures(reader);
  if (!signatures.ok()) return signatures.status();
  if (!reader.done()) return invalid_encoding();
  return SealedObject{*signed_bytes, std::move(*signatures)};
}

bool exact(std::span<const std::byte> left,
           std::span<const std::byte> right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
}

bool starts_with_domain(std::span<const std::byte> bytes,
                        std::string_view domain) {
  const auto expected = std::as_bytes(std::span(domain));
  return bytes.size() >= expected.size() + 1 &&
         std::equal(expected.begin(), expected.end(), bytes.begin()) &&
         bytes[expected.size()] == std::byte{0};
}

}  // namespace

Result<ProfileTrustPolicy> parse_profile_trust_policy(
    std::span<const std::byte> bytes) {
  if (bytes.empty() || bytes.size() > kProfileCatalogMaximumBytes) {
    return invalid_encoding();
  }
  Reader reader(bytes);
  auto status = reader.domain("pih:profile-trust-policy:v1");
  if (!status.ok()) return status;
  auto generation = reader.u64();
  auto has_root = reader.u8();
  if (!generation.ok() || !has_root.ok() || *has_root > 1) {
    return invalid_encoding();
  }
  std::optional<Sha256Digest> exact_root;
  if (*has_root == 1) {
    auto root = reader.fixed(32);
    if (!root.ok()) return root.status();
    exact_root.emplace();
    std::copy(root->begin(), root->end(), exact_root->bytes.begin());
  }
  auto count = reader.u32();
  if (!count.ok() || *count == 0 || *count > kProfileTrustMaximumKeys) {
    return invalid_encoding();
  }
  std::vector<ProfileTrustKey> keys;
  keys.reserve(*count);
  for (std::uint32_t index = 0; index < *count; ++index) {
    auto key_id = reader.text();
    auto role = reader.u8();
    auto revoked = reader.u8();
    auto public_key = reader.fixed(32);
    if (!key_id.ok() || !role.ok() || !revoked.ok() || !public_key.ok() ||
        *revoked > 1) {
      return invalid_encoding();
    }
    ProfileTrustKey key{};
    key.key_id = std::move(*key_id);
    key.role = static_cast<ProfileSignerRole>(*role);
    key.revoked = *revoked == 1;
    std::copy(public_key->begin(), public_key->end(), key.public_key.begin());
    keys.push_back(std::move(key));
  }
  if (!reader.done()) return invalid_encoding();
  auto result = ProfileTrustPolicy::Create(*generation, exact_root, std::move(keys));
  if (!result.ok()) return result.status();
  if (!exact(result->canonical_object_bytes(), bytes)) return invalid_encoding();
  return result;
}

Result<SignedProfileEnvelope> parse_signed_profile_envelope(
    std::span<const std::byte> bytes) {
  if (bytes.empty() || bytes.size() > kProfileEnvelopeMaximumBytes) {
    return invalid_encoding();
  }
  auto sealed = parse_sealed(bytes, "pih:profile-envelope-object:v1",
                             kProfileEnvelopeMaximumBytes);
  if (!sealed.ok()) return sealed.status();
  Reader reader(sealed->signed_bytes);
  auto status = reader.domain("pih:signed-profile-envelope:v1");
  if (!status.ok()) return status;
  auto profile_id = reader.text();
  auto revision = reader.text();
  auto generation = reader.u64();
  auto payload = reader.blob(kProfilePayloadMaximumBytes);
  auto closure = reader.fixed(32);
  if (!profile_id.ok() || !revision.ok() || !generation.ok() || !payload.ok() ||
      !closure.ok() || !reader.done()) {
    return invalid_encoding();
  }
  Sha256Digest closure_root{};
  std::copy(closure->begin(), closure->end(), closure_root.bytes.begin());
  auto result = SignedProfileEnvelope::Create(
      std::move(*profile_id), std::move(*revision), *generation, *payload,
      closure_root, std::move(sealed->signatures));
  if (!result.ok()) return result.status();
  if (!exact(result->canonical_object_bytes(), bytes)) return invalid_encoding();
  return result;
}

Result<SignedProfileCatalog> parse_signed_profile_catalog(
    std::span<const std::byte> bytes) {
  if (bytes.empty() || bytes.size() > kProfileCatalogMaximumBytes) {
    return invalid_encoding();
  }
  const bool graph_bound = starts_with_domain(
      bytes, "pih:profile-catalog-object:v2");
  const auto outer_domain = graph_bound
      ? "pih:profile-catalog-object:v2"
      : "pih:profile-catalog-object:v1";
  auto sealed = parse_sealed(bytes, outer_domain,
                             kProfileCatalogMaximumBytes);
  if (!sealed.ok()) return sealed.status();
  Reader reader(sealed->signed_bytes);
  auto status = reader.domain(
      graph_bound ? "pih:immutable-profile-catalog:v2"
                  : "pih:immutable-profile-catalog:v1");
  if (!status.ok()) return status;
  auto generation = reader.u64();
  ProfileCatalogGraphBinding graph_binding{};
  if (graph_bound) {
    auto graph_root = reader.fixed(32);
    auto node_count = reader.u32();
    auto edge_count = reader.u32();
    if (!graph_root.ok() || !node_count.ok() || !edge_count.ok()) {
      return invalid_encoding();
    }
    std::copy(graph_root->begin(), graph_root->end(),
              graph_binding.snapshot_root.bytes.begin());
    graph_binding.node_count = *node_count;
    graph_binding.edge_count = *edge_count;
  }
  auto count = reader.u32();
  if (!generation.ok() || !count.ok() || *count == 0 ||
      *count > kProfileCatalogMaximumEntries) {
    return invalid_encoding();
  }
  std::vector<ProfileCatalogEntry> entries;
  entries.reserve(*count);
  for (std::uint32_t index = 0; index < *count; ++index) {
    auto profile_id = reader.text();
    auto revision = reader.text();
    auto root = reader.fixed(32);
    auto envelope_bytes = reader.u64();
    if (!profile_id.ok() || !revision.ok() || !root.ok() ||
        !envelope_bytes.ok()) {
      return invalid_encoding();
    }
    ProfileCatalogEntry entry{};
    entry.profile_id = std::move(*profile_id);
    entry.revision = std::move(*revision);
    std::copy(root->begin(), root->end(), entry.envelope_root.bytes.begin());
    entry.envelope_bytes = *envelope_bytes;
    entries.push_back(std::move(entry));
  }
  if (!reader.done()) return invalid_encoding();
  auto result = graph_bound
      ? SignedProfileCatalog::CreateGraphBound(
            *generation, graph_binding, std::move(entries),
            std::move(sealed->signatures))
      : SignedProfileCatalog::Create(
            *generation, std::move(entries), std::move(sealed->signatures));
  if (!result.ok()) return result.status();
  if (!exact(result->canonical_object_bytes(), bytes)) return invalid_encoding();
  return result;
}

}  // namespace pih
