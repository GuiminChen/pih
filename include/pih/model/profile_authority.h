#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pih/core/sha256.h"

namespace pih {

inline constexpr std::size_t kProfileEnvelopeMaximumBytes = 16'777'216;
inline constexpr std::size_t kProfilePayloadMaximumBytes = 15'728'640;
inline constexpr std::size_t kProfileCatalogMaximumBytes = 4'194'304;
inline constexpr std::size_t kProfileCatalogMaximumEntries = 4'096;
inline constexpr std::size_t kProfileTrustMaximumKeys = 32;
inline constexpr std::size_t kProfileMaximumSignatures = 8;
inline constexpr std::size_t kEd25519SignatureBytes = 64;

enum class ProfileSignerRole : std::uint8_t {
  kRelease = 1,
  kBuilder = 2,
};

struct ProfileTrustKey final {
  std::string key_id;
  ProfileSignerRole role = ProfileSignerRole::kBuilder;
  bool revoked = false;
  std::array<std::byte, 32> public_key{};
};

struct ProfileSignature final {
  std::string key_id;
  std::vector<std::byte> bytes;
};

class ProfileTrustPolicy final {
 public:
  static Result<ProfileTrustPolicy> Create(
      std::uint64_t minimum_catalog_generation,
      std::optional<Sha256Digest> exact_catalog_root,
      std::vector<ProfileTrustKey> keys);

  [[nodiscard]] const Sha256Digest& policy_digest() const noexcept {
    return policy_digest_;
  }
  [[nodiscard]] std::uint64_t minimum_catalog_generation() const noexcept {
    return minimum_catalog_generation_;
  }
  [[nodiscard]] const std::optional<Sha256Digest>& exact_catalog_root()
      const noexcept {
    return exact_catalog_root_;
  }
  [[nodiscard]] const ProfileTrustKey* key(std::string_view key_id) const noexcept;
  [[nodiscard]] std::span<const std::byte> canonical_object_bytes() const noexcept {
    return canonical_object_bytes_;
  }

 private:
  ProfileTrustPolicy(Sha256Digest policy_digest,
                     std::uint64_t minimum_catalog_generation,
                     std::optional<Sha256Digest> exact_catalog_root,
                     std::vector<ProfileTrustKey> keys,
                     std::vector<std::byte> canonical_object_bytes) noexcept
      : policy_digest_(policy_digest),
        minimum_catalog_generation_(minimum_catalog_generation),
        exact_catalog_root_(exact_catalog_root), keys_(std::move(keys)),
        canonical_object_bytes_(std::move(canonical_object_bytes)) {}

  Sha256Digest policy_digest_{};
  std::uint64_t minimum_catalog_generation_ = 0;
  std::optional<Sha256Digest> exact_catalog_root_;
  std::vector<ProfileTrustKey> keys_;
  std::vector<std::byte> canonical_object_bytes_;
};

class SignedProfileEnvelope final {
 public:
  static Result<SignedProfileEnvelope> Create(
      std::string profile_id, std::string revision,
      std::uint64_t catalog_generation,
      std::span<const std::byte> payload_bytes,
      Sha256Digest reference_closure_root,
      std::vector<ProfileSignature> signatures);

  [[nodiscard]] std::string_view profile_id() const noexcept {
    return profile_id_;
  }
  [[nodiscard]] std::string_view revision() const noexcept { return revision_; }
  [[nodiscard]] std::uint64_t catalog_generation() const noexcept {
    return catalog_generation_;
  }
  [[nodiscard]] const Sha256Digest& payload_root() const noexcept {
    return payload_root_;
  }
  [[nodiscard]] std::span<const std::byte> payload_bytes() const noexcept {
    return std::span<const std::byte>(canonical_signed_bytes_)
        .subspan(payload_offset_, payload_size_);
  }
  [[nodiscard]] const Sha256Digest& reference_closure_root() const noexcept {
    return reference_closure_root_;
  }
  [[nodiscard]] const Sha256Digest& envelope_root() const noexcept {
    return envelope_root_;
  }
  [[nodiscard]] std::uint64_t envelope_bytes() const noexcept {
    return envelope_bytes_;
  }
  [[nodiscard]] std::span<const std::byte> canonical_signed_bytes() const noexcept {
    return canonical_signed_bytes_;
  }
  [[nodiscard]] std::span<const ProfileSignature> signatures() const noexcept {
    return signatures_;
  }
  [[nodiscard]] std::span<const std::byte> canonical_object_bytes() const noexcept {
    return canonical_object_bytes_;
  }

 private:
  SignedProfileEnvelope(
      std::string profile_id, std::string revision,
      std::uint64_t catalog_generation, Sha256Digest payload_root,
      Sha256Digest reference_closure_root, Sha256Digest envelope_root,
      std::uint64_t envelope_bytes,
      std::vector<std::byte> canonical_signed_bytes,
      std::vector<ProfileSignature> signatures,
      std::vector<std::byte> canonical_object_bytes,
      std::size_t payload_offset, std::size_t payload_size) noexcept
      : profile_id_(std::move(profile_id)), revision_(std::move(revision)),
        catalog_generation_(catalog_generation), payload_root_(payload_root),
        reference_closure_root_(reference_closure_root),
        envelope_root_(envelope_root), envelope_bytes_(envelope_bytes),
        canonical_signed_bytes_(std::move(canonical_signed_bytes)),
        signatures_(std::move(signatures)),
        canonical_object_bytes_(std::move(canonical_object_bytes)),
        payload_offset_(payload_offset), payload_size_(payload_size) {}

  std::string profile_id_;
  std::string revision_;
  std::uint64_t catalog_generation_ = 0;
  Sha256Digest payload_root_{};
  Sha256Digest reference_closure_root_{};
  Sha256Digest envelope_root_{};
  std::uint64_t envelope_bytes_ = 0;
  std::vector<std::byte> canonical_signed_bytes_;
  std::vector<ProfileSignature> signatures_;
  std::vector<std::byte> canonical_object_bytes_;
  std::size_t payload_offset_ = 0;
  std::size_t payload_size_ = 0;
};

struct ProfileCatalogEntry final {
  std::string profile_id;
  std::string revision;
  Sha256Digest envelope_root{};
  std::uint64_t envelope_bytes = 0;
};

struct ProfileCatalogGraphBinding final {
  Sha256Digest snapshot_root{};
  std::uint32_t node_count = 0;
  std::uint32_t edge_count = 0;
};

class SignedProfileCatalog final {
 public:
  static Result<SignedProfileCatalog> Create(
      std::uint64_t generation,
      std::vector<ProfileCatalogEntry> entries,
      std::vector<ProfileSignature> signatures);
  static Result<SignedProfileCatalog> CreateGraphBound(
      std::uint64_t generation,
      ProfileCatalogGraphBinding graph_binding,
      std::vector<ProfileCatalogEntry> entries,
      std::vector<ProfileSignature> signatures);

  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] const Sha256Digest& catalog_root() const noexcept {
    return catalog_root_;
  }
  [[nodiscard]] bool graph_bound() const noexcept {
    return graph_binding_.has_value();
  }
  [[nodiscard]] const std::optional<ProfileCatalogGraphBinding>& graph_binding()
      const noexcept { return graph_binding_; }
  [[nodiscard]] const ProfileCatalogEntry* find(
      std::string_view profile_id, std::string_view revision) const noexcept;
  [[nodiscard]] std::span<const std::byte> canonical_signed_bytes() const noexcept {
    return canonical_signed_bytes_;
  }
  [[nodiscard]] std::span<const ProfileSignature> signatures() const noexcept {
    return signatures_;
  }
  [[nodiscard]] std::span<const std::byte> canonical_object_bytes() const noexcept {
    return canonical_object_bytes_;
  }

 private:
  static Result<SignedProfileCatalog> CreateImpl(
      std::uint64_t generation,
      std::optional<ProfileCatalogGraphBinding> graph_binding,
      std::vector<ProfileCatalogEntry> entries,
      std::vector<ProfileSignature> signatures);
  SignedProfileCatalog(std::uint64_t generation,
                       std::optional<ProfileCatalogGraphBinding> graph_binding,
                       Sha256Digest catalog_root,
                       std::vector<std::byte> canonical_signed_bytes,
                       std::vector<ProfileCatalogEntry> entries,
                       std::vector<ProfileSignature> signatures,
                       std::vector<std::byte> canonical_object_bytes) noexcept
      : generation_(generation), graph_binding_(graph_binding),
        catalog_root_(catalog_root),
        canonical_signed_bytes_(std::move(canonical_signed_bytes)),
        entries_(std::move(entries)), signatures_(std::move(signatures)),
        canonical_object_bytes_(std::move(canonical_object_bytes)) {}

  std::uint64_t generation_ = 0;
  std::optional<ProfileCatalogGraphBinding> graph_binding_;
  Sha256Digest catalog_root_{};
  std::vector<std::byte> canonical_signed_bytes_;
  std::vector<ProfileCatalogEntry> entries_;
  std::vector<ProfileSignature> signatures_;
  std::vector<std::byte> canonical_object_bytes_;
};

class ProfileEd25519Verifier {
 public:
  virtual ~ProfileEd25519Verifier() = default;
  virtual Status verify(std::string_view key_id,
                        std::span<const std::byte> public_key,
                        std::span<const std::byte> message,
                        std::span<const std::byte> signature) = 0;
};

class VerifiedProfileAuthority final {
 public:
  [[nodiscard]] bool verified() const noexcept { return true; }
  [[nodiscard]] const Sha256Digest& policy_digest() const noexcept {
    return policy_digest_;
  }
  [[nodiscard]] std::string_view profile_id() const noexcept {
    return profile_id_;
  }
  [[nodiscard]] std::string_view revision() const noexcept {
    return revision_;
  }
  [[nodiscard]] std::uint64_t catalog_generation() const noexcept {
    return catalog_generation_;
  }
  [[nodiscard]] const Sha256Digest& catalog_root() const noexcept {
    return catalog_root_;
  }
  [[nodiscard]] const Sha256Digest& envelope_root() const noexcept {
    return envelope_root_;
  }
  [[nodiscard]] const Sha256Digest& payload_root() const noexcept {
    return payload_root_;
  }
  [[nodiscard]] const Sha256Digest& reference_closure_root() const noexcept {
    return reference_closure_root_;
  }

 private:
  friend Result<VerifiedProfileAuthority> verify_profile_authority(
      const ProfileTrustPolicy&, const SignedProfileCatalog&,
      const SignedProfileEnvelope&, std::string_view, std::string_view,
      ProfileEd25519Verifier&);
  VerifiedProfileAuthority(std::string profile_id, std::string revision,
                           Sha256Digest policy_digest,
                           std::uint64_t catalog_generation,
                           Sha256Digest catalog_root,
                           Sha256Digest envelope_root,
                           Sha256Digest payload_root,
                           Sha256Digest reference_closure_root) noexcept
      : profile_id_(std::move(profile_id)), revision_(std::move(revision)),
        policy_digest_(policy_digest), catalog_generation_(catalog_generation),
        catalog_root_(catalog_root),
        envelope_root_(envelope_root), payload_root_(payload_root),
        reference_closure_root_(reference_closure_root) {}

  std::string profile_id_;
  std::string revision_;
  Sha256Digest policy_digest_{};
  std::uint64_t catalog_generation_ = 0;
  Sha256Digest catalog_root_{};
  Sha256Digest envelope_root_{};
  Sha256Digest payload_root_{};
  Sha256Digest reference_closure_root_{};
};

Result<VerifiedProfileAuthority> verify_profile_authority(
    const ProfileTrustPolicy& policy, const SignedProfileCatalog& catalog,
    const SignedProfileEnvelope& envelope, std::string_view requested_profile_id,
    std::string_view requested_revision, ProfileEd25519Verifier& verifier);

}  // namespace pih
