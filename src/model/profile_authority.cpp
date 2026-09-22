#include "pih/model/profile_authority.h"

#include "pih/model/sealed_reachable_object_dag.h"

#include <algorithm>
#include <unordered_set>

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) {
  return std::any_of(value.bytes.begin(), value.bytes.end(),
                     [](std::byte byte) { return byte != std::byte{0}; });
}

bool valid_text(std::string_view value) {
  if (value.empty() || value.size() > 128) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char byte) {
    return byte >= 0x21 && byte <= 0x7e;
  });
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
  for (std::size_t index = 0; index < 4; ++index) {
    output.push_back(static_cast<std::byte>(value >> (index * 8U)));
  }
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::size_t index = 0; index < 8; ++index) {
    output.push_back(static_cast<std::byte>(value >> (index * 8U)));
  }
}

void append_bytes(std::vector<std::byte>& output,
                  std::span<const std::byte> value) {
  append_u64(output, value.size());
  output.insert(output.end(), value.begin(), value.end());
}

void append_text(std::vector<std::byte>& output, std::string_view value) {
  append_bytes(output, std::as_bytes(std::span(value)));
}

std::vector<std::byte> begin_object(std::string_view domain) {
  std::vector<std::byte> output;
  output.reserve(domain.size() + 1);
  const auto bytes = std::as_bytes(std::span(domain));
  output.insert(output.end(), bytes.begin(), bytes.end());
  output.push_back(std::byte{0});
  return output;
}

std::vector<std::byte> seal_signed_object(
    std::string_view domain, std::span<const std::byte> signed_bytes,
    std::span<const ProfileSignature> signatures) {
  auto output = begin_object(domain);
  append_bytes(output, signed_bytes);
  append_u32(output, static_cast<std::uint32_t>(signatures.size()));
  for (const auto& signature : signatures) {
    append_text(output, signature.key_id);
    append_bytes(output, signature.bytes);
  }
  return output;
}

Status canonicalize_signatures(std::vector<ProfileSignature>& signatures) {
  if (signatures.empty() || signatures.size() > kProfileMaximumSignatures) {
    return Status::InvalidArgument("profile signature count is invalid");
  }
  for (const auto& signature : signatures) {
    if (!valid_text(signature.key_id) ||
        signature.bytes.size() != kEd25519SignatureBytes) {
      return Status::InvalidArgument("profile Ed25519 signature is invalid");
    }
  }
  std::stable_sort(signatures.begin(), signatures.end(),
                   [](const auto& left, const auto& right) {
                     return left.key_id < right.key_id;
                   });
  for (std::size_t index = 1; index < signatures.size(); ++index) {
    if (signatures[index - 1].key_id == signatures[index].key_id &&
        signatures[index - 1].bytes != signatures[index].bytes) {
      return Status::InvalidArgument(
          "profile signatures for one key are ambiguous");
    }
  }
  return Status::Ok();
}

Status verify_release_signature(
    const ProfileTrustPolicy& policy,
    std::span<const ProfileSignature> signatures,
    std::span<const std::byte> message, ProfileEd25519Verifier& verifier) {
  bool eligible = false;
  std::string_view previous_key;
  for (const auto& signature : signatures) {
    if (signature.key_id == previous_key) continue;
    previous_key = signature.key_id;
    const auto* key = policy.key(signature.key_id);
    if (key == nullptr) {
      return Status::FailedPrecondition("profile signature key is not trusted");
    }
    if (key->revoked || key->role != ProfileSignerRole::kRelease) continue;
    eligible = true;
    if (verifier.verify(signature.key_id, key->public_key, message,
                        signature.bytes).ok()) {
      return Status::Ok();
    }
  }
  return Status::FailedPrecondition(
      eligible ? "profile release signature verification failed"
               : "profile has no eligible release signature");
}

}  // namespace

Result<ProfileTrustPolicy> ProfileTrustPolicy::Create(
    std::uint64_t minimum_catalog_generation,
    std::optional<Sha256Digest> exact_catalog_root,
    std::vector<ProfileTrustKey> keys) {
  if (minimum_catalog_generation == 0 ||
      keys.empty() || keys.size() > kProfileTrustMaximumKeys ||
      (exact_catalog_root.has_value() && !nonzero(*exact_catalog_root))) {
    return Status::InvalidArgument("profile trust policy is invalid");
  }
  std::unordered_set<std::string> identities;
  identities.reserve(keys.size());
  for (const auto& key : keys) {
    if (!valid_text(key.key_id) || !nonzero(Sha256Digest{key.public_key}) ||
        (key.role != ProfileSignerRole::kRelease &&
         key.role != ProfileSignerRole::kBuilder) ||
        !identities.insert(key.key_id).second) {
      return Status::InvalidArgument("profile trust key set is invalid");
    }
  }
  std::sort(keys.begin(), keys.end(),
            [](const auto& left, const auto& right) {
              return left.key_id < right.key_id;
            });
  auto canonical = begin_object("pih:profile-trust-policy:v1");
  append_u64(canonical, minimum_catalog_generation);
  canonical.push_back(exact_catalog_root.has_value() ? std::byte{1}
                                                     : std::byte{0});
  if (exact_catalog_root.has_value()) {
    canonical.insert(canonical.end(), exact_catalog_root->bytes.begin(),
                     exact_catalog_root->bytes.end());
  }
  append_u32(canonical, static_cast<std::uint32_t>(keys.size()));
  for (const auto& key : keys) {
    append_text(canonical, key.key_id);
    canonical.push_back(static_cast<std::byte>(key.role));
    canonical.push_back(key.revoked ? std::byte{1} : std::byte{0});
    canonical.insert(canonical.end(), key.public_key.begin(),
                     key.public_key.end());
  }
  auto policy_digest = sha256(canonical);
  if (!policy_digest.ok()) return policy_digest.status();
  return ProfileTrustPolicy(*policy_digest, minimum_catalog_generation,
                            exact_catalog_root, std::move(keys),
                            std::move(canonical));
}

const ProfileTrustKey* ProfileTrustPolicy::key(
    std::string_view key_id) const noexcept {
  for (const auto& key : keys_) {
    if (key.key_id == key_id) return &key;
  }
  return nullptr;
}

Result<SignedProfileEnvelope> SignedProfileEnvelope::Create(
    std::string profile_id, std::string revision,
    std::uint64_t catalog_generation,
    std::span<const std::byte> payload_bytes,
    Sha256Digest reference_closure_root,
    std::vector<ProfileSignature> signatures) {
  if (!valid_text(profile_id) || !valid_text(revision) ||
      catalog_generation == 0 || payload_bytes.empty() ||
      payload_bytes.size() > kProfilePayloadMaximumBytes ||
      !nonzero(reference_closure_root)) {
    return Status::InvalidArgument("signed profile envelope is invalid");
  }
  const auto signature_status = canonicalize_signatures(signatures);
  if (!signature_status.ok()) return signature_status;
  auto payload_root = sha256(payload_bytes);
  if (!payload_root.ok()) return payload_root.status();
  auto signed_bytes = begin_object("pih:signed-profile-envelope:v1");
  append_text(signed_bytes, profile_id);
  append_text(signed_bytes, revision);
  append_u64(signed_bytes, catalog_generation);
  const auto payload_offset = signed_bytes.size() + sizeof(std::uint64_t);
  append_bytes(signed_bytes, payload_bytes);
  signed_bytes.insert(signed_bytes.end(), reference_closure_root.bytes.begin(),
                      reference_closure_root.bytes.end());
  auto envelope_bytes = seal_signed_object(
      "pih:profile-envelope-object:v1", signed_bytes, signatures);
  if (envelope_bytes.size() > kProfileEnvelopeMaximumBytes) {
    return Status::ResourceExhausted("signed profile envelope exceeds bound");
  }
  auto envelope_root = sha256(envelope_bytes);
  if (!envelope_root.ok()) return envelope_root.status();
  const auto envelope_size = envelope_bytes.size();
  return SignedProfileEnvelope(
      std::move(profile_id), std::move(revision), catalog_generation,
      *payload_root, reference_closure_root, *envelope_root,
      envelope_size, std::move(signed_bytes),
      std::move(signatures), std::move(envelope_bytes), payload_offset,
      payload_bytes.size());
}

Result<SignedProfileCatalog> SignedProfileCatalog::Create(
    std::uint64_t generation,
    std::vector<ProfileCatalogEntry> entries,
    std::vector<ProfileSignature> signatures) {
  return CreateImpl(generation, std::nullopt, std::move(entries),
                    std::move(signatures));
}

Result<SignedProfileCatalog> SignedProfileCatalog::CreateGraphBound(
    std::uint64_t generation,
    ProfileCatalogGraphBinding graph_binding,
    std::vector<ProfileCatalogEntry> entries,
    std::vector<ProfileSignature> signatures) {
  if (!nonzero(graph_binding.snapshot_root) || graph_binding.node_count == 0 ||
      graph_binding.node_count > kDigestDagMaximumNodes ||
      graph_binding.edge_count > kDigestDagMaximumEdges) {
    return Status::InvalidArgument("profile catalog graph binding is invalid");
  }
  return CreateImpl(generation, graph_binding, std::move(entries),
                    std::move(signatures));
}

Result<SignedProfileCatalog> SignedProfileCatalog::CreateImpl(
    std::uint64_t generation,
    std::optional<ProfileCatalogGraphBinding> graph_binding,
    std::vector<ProfileCatalogEntry> entries,
    std::vector<ProfileSignature> signatures) {
  if (generation == 0 || entries.empty() ||
      entries.size() > kProfileCatalogMaximumEntries) {
    return Status::InvalidArgument("signed profile catalog is invalid");
  }
  const auto signature_status = canonicalize_signatures(signatures);
  if (!signature_status.ok()) return signature_status;
  for (const auto& entry : entries) {
    if (!valid_text(entry.profile_id) || !valid_text(entry.revision) ||
        !nonzero(entry.envelope_root) || entry.envelope_bytes == 0 ||
        entry.envelope_bytes > kProfileEnvelopeMaximumBytes) {
      return Status::InvalidArgument("profile catalog entry is invalid");
    }
  }
  std::sort(entries.begin(), entries.end(), [](const auto& left,
                                                const auto& right) {
    return std::tie(left.profile_id, left.revision) <
           std::tie(right.profile_id, right.revision);
  });
  for (std::size_t index = 1; index < entries.size(); ++index) {
    if (entries[index - 1].profile_id == entries[index].profile_id &&
        entries[index - 1].revision == entries[index].revision) {
      return Status::InvalidArgument("profile catalog identity is duplicated");
    }
  }
  auto signed_bytes = begin_object(
      graph_binding.has_value()
          ? "pih:immutable-profile-catalog:v2"
          : "pih:immutable-profile-catalog:v1");
  append_u64(signed_bytes, generation);
  if (graph_binding.has_value()) {
    signed_bytes.insert(signed_bytes.end(),
                        graph_binding->snapshot_root.bytes.begin(),
                        graph_binding->snapshot_root.bytes.end());
    append_u32(signed_bytes, graph_binding->node_count);
    append_u32(signed_bytes, graph_binding->edge_count);
  }
  append_u32(signed_bytes, static_cast<std::uint32_t>(entries.size()));
  for (const auto& entry : entries) {
    append_text(signed_bytes, entry.profile_id);
    append_text(signed_bytes, entry.revision);
    signed_bytes.insert(signed_bytes.end(), entry.envelope_root.bytes.begin(),
                        entry.envelope_root.bytes.end());
    append_u64(signed_bytes, entry.envelope_bytes);
  }
  auto catalog_bytes = seal_signed_object(
      graph_binding.has_value()
          ? "pih:profile-catalog-object:v2"
          : "pih:profile-catalog-object:v1",
      signed_bytes, signatures);
  if (catalog_bytes.size() > kProfileCatalogMaximumBytes) {
    return Status::ResourceExhausted("signed profile catalog exceeds bound");
  }
  auto catalog_root = sha256(catalog_bytes);
  if (!catalog_root.ok()) return catalog_root.status();
  return SignedProfileCatalog(
      generation, graph_binding, *catalog_root,
      std::move(signed_bytes),
      std::move(entries), std::move(signatures), std::move(catalog_bytes));
}

const ProfileCatalogEntry* SignedProfileCatalog::find(
    std::string_view profile_id, std::string_view revision) const noexcept {
  for (const auto& entry : entries_) {
    if (entry.profile_id == profile_id && entry.revision == revision) {
      return &entry;
    }
  }
  return nullptr;
}

Result<VerifiedProfileAuthority> verify_profile_authority(
    const ProfileTrustPolicy& policy, const SignedProfileCatalog& catalog,
    const SignedProfileEnvelope& envelope,
    std::string_view requested_profile_id,
    std::string_view requested_revision, ProfileEd25519Verifier& verifier) {
  if (!valid_text(requested_profile_id) || !valid_text(requested_revision)) {
    return Status::InvalidArgument("requested exact profile identity is invalid");
  }
  if (catalog.generation() < policy.minimum_catalog_generation() ||
      (policy.exact_catalog_root().has_value() &&
       !(*policy.exact_catalog_root() == catalog.catalog_root()))) {
    return Status::FailedPrecondition(
        "profile catalog violates deployment trust policy");
  }
  auto verified = verify_release_signature(
      policy, catalog.signatures(), catalog.canonical_signed_bytes(), verifier);
  if (!verified.ok()) return verified;

  const auto* entry = catalog.find(requested_profile_id, requested_revision);
  if (entry == nullptr || envelope.profile_id() != requested_profile_id ||
      envelope.revision() != requested_revision ||
      envelope.catalog_generation() != catalog.generation() ||
      !(entry->envelope_root == envelope.envelope_root()) ||
      entry->envelope_bytes != envelope.envelope_bytes()) {
    return Status::FailedPrecondition(
        "profile envelope is not the exact catalog member");
  }
  verified = verify_release_signature(
      policy, envelope.signatures(), envelope.canonical_signed_bytes(), verifier);
  if (!verified.ok()) return verified;
  return VerifiedProfileAuthority(
      std::string(requested_profile_id), std::string(requested_revision),
      policy.policy_digest(), catalog.generation(), catalog.catalog_root(),
      envelope.envelope_root(),
      envelope.payload_root(), envelope.reference_closure_root());
}

}  // namespace pih
