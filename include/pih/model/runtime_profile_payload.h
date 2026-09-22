#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "pih/model/profile_authority.h"

namespace pih {

class VerifiedRuntimeProfileDevices;
class RuntimeProfileReferenceClosure;
class VerifiedRuntimeProfileReferenceLeases;
class VerifiedRuntimeProfileAuthorityLeases;
class RuntimeEvidenceProjection;
class RuntimeEngineAdmission;

enum class RuntimeProfileReferenceRetention : std::uint8_t {
  kNone = 0,
  kMemorySnapshot = 1,
  kImmutableDescriptorLease = 2,
};

Result<RuntimeEngineAdmission> retain_runtime_engine_reference_objects(
    RuntimeEngineAdmission admission,
    const RuntimeProfileReferenceClosure& closure,
    std::vector<std::vector<std::byte>> objects);
Result<RuntimeEngineAdmission> retain_runtime_engine_reference_leases(
    RuntimeEngineAdmission admission,
    VerifiedRuntimeProfileReferenceLeases leases);
Result<RuntimeEngineAdmission> retain_runtime_engine_authority_leases(
    RuntimeEngineAdmission admission,
    VerifiedRuntimeProfileAuthorityLeases leases);

enum class RuntimeProfileModel : std::uint8_t {
  kQwen3_0_6B = 1,
  kDeepSeekV4Flash0731 = 2,
};

enum class RuntimeProfileWeightFormat : std::uint8_t {
  kBf16 = 1,
  kXingInt4 = 2,
  kDeepSeekNative = 3,
};

enum class RuntimeProfileGpuFamily : std::uint8_t {
  kRtx4090D24GiB = 1,
  kH100Pcie80GiB = 2,
};

enum class RuntimeProfileResidency : std::uint8_t {
  kFullResident = 1,
  kHostSpill = 2,
};

enum class RuntimeProfileEvidenceState : std::uint8_t {
  kHardwareEvidenceOpen = 1,
  kCorrectnessSupported = 2,
};

struct RuntimeProfileRoots final {
  Sha256Digest runtime_semantic_root{};
  Sha256Digest capacity_template_root{};
  Sha256Digest feature_selection_root{};
  Sha256Digest release_evidence_root{};
  Sha256Digest hardware_identity_root{};
  Sha256Digest kernel_closure_root{};
  Sha256Digest security_runtime_root{};
};

struct RuntimeEvidenceRoots final {
  Sha256Digest bundle_root{};
  Sha256Digest audit_closure_root{};
  Sha256Digest required_role_schema_root{};
  Sha256Digest domain_decision_roots_digest{};
  Sha256Digest builder_policy_root{};
};

class RuntimeProfilePayload final {
 public:
  static Result<RuntimeProfilePayload> Create(
      RuntimeProfileModel model, RuntimeProfileWeightFormat weight_format,
      RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
      bool dspark_enabled, RuntimeProfileResidency residency,
      RuntimeProfileEvidenceState evidence_state, RuntimeProfileRoots roots);

  [[nodiscard]] RuntimeProfileModel model() const noexcept { return model_; }
  [[nodiscard]] RuntimeProfileWeightFormat weight_format() const noexcept {
    return weight_format_;
  }
  [[nodiscard]] RuntimeProfileGpuFamily gpu_family() const noexcept {
    return gpu_family_;
  }
  [[nodiscard]] std::uint8_t world_size() const noexcept { return world_size_; }
  [[nodiscard]] bool dspark_enabled() const noexcept { return dspark_enabled_; }
  [[nodiscard]] RuntimeProfileResidency residency() const noexcept {
    return residency_;
  }
  [[nodiscard]] RuntimeProfileEvidenceState evidence_state() const noexcept {
    return evidence_state_;
  }
  [[nodiscard]] const RuntimeProfileRoots& roots() const noexcept {
    return roots_;
  }
  [[nodiscard]] std::span<const std::byte> canonical_bytes() const noexcept {
    return canonical_bytes_;
  }

 private:
  RuntimeProfilePayload(RuntimeProfileModel model,
                        RuntimeProfileWeightFormat weight_format,
                        RuntimeProfileGpuFamily gpu_family,
                        std::uint8_t world_size, bool dspark_enabled,
                        RuntimeProfileResidency residency,
                        RuntimeProfileEvidenceState evidence_state,
                        RuntimeProfileRoots roots,
                        std::vector<std::byte> canonical_bytes) noexcept;

  RuntimeProfileModel model_ = RuntimeProfileModel::kQwen3_0_6B;
  RuntimeProfileWeightFormat weight_format_ = RuntimeProfileWeightFormat::kBf16;
  RuntimeProfileGpuFamily gpu_family_ = RuntimeProfileGpuFamily::kRtx4090D24GiB;
  std::uint8_t world_size_ = 0;
  bool dspark_enabled_ = false;
  RuntimeProfileResidency residency_ = RuntimeProfileResidency::kFullResident;
  RuntimeProfileEvidenceState evidence_state_ =
      RuntimeProfileEvidenceState::kHardwareEvidenceOpen;
  RuntimeProfileRoots roots_{};
  std::vector<std::byte> canonical_bytes_;
};

Result<RuntimeProfilePayload> parse_runtime_profile_payload(
    std::span<const std::byte> bytes);

class VerifiedRuntimeProfile final {
 public:
  [[nodiscard]] std::string_view profile_id() const noexcept {
    return profile_id_;
  }
  [[nodiscard]] std::string_view revision() const noexcept { return revision_; }
  [[nodiscard]] const RuntimeProfilePayload& payload() const noexcept {
    return payload_;
  }
  [[nodiscard]] const Sha256Digest& envelope_root() const noexcept {
    return envelope_root_;
  }
  [[nodiscard]] const Sha256Digest& reference_closure_root() const noexcept {
    return reference_closure_root_;
  }
  [[nodiscard]] const Sha256Digest& policy_digest() const noexcept {
    return policy_digest_;
  }
  [[nodiscard]] const Sha256Digest& catalog_root() const noexcept {
    return catalog_root_;
  }
 private:
  friend Result<VerifiedRuntimeProfile> bind_verified_runtime_profile(
      const VerifiedProfileAuthority&, const SignedProfileEnvelope&);
  VerifiedRuntimeProfile(std::string profile_id, std::string revision,
                         Sha256Digest policy_digest,
                         Sha256Digest catalog_root,
                         Sha256Digest envelope_root,
                         Sha256Digest reference_closure_root,
                         RuntimeProfilePayload payload) noexcept;
  std::string profile_id_;
  std::string revision_;
  Sha256Digest policy_digest_{};
  Sha256Digest catalog_root_{};
  Sha256Digest envelope_root_{};
  Sha256Digest reference_closure_root_{};
  RuntimeProfilePayload payload_;
};

Result<VerifiedRuntimeProfile> bind_verified_runtime_profile(
    const VerifiedProfileAuthority& authority,
    const SignedProfileEnvelope& envelope);

class RuntimeEngineAdmission final {
 public:
  [[nodiscard]] std::string_view profile_id() const noexcept {
    return profile_id_;
  }
  [[nodiscard]] std::string_view revision() const noexcept { return revision_; }
  [[nodiscard]] const Sha256Digest& envelope_root() const noexcept {
    return envelope_root_;
  }
  [[nodiscard]] const Sha256Digest& reference_closure_root() const noexcept {
    return reference_closure_root_;
  }
  [[nodiscard]] const Sha256Digest& policy_digest() const noexcept {
    return policy_digest_;
  }
  [[nodiscard]] const Sha256Digest& catalog_root() const noexcept {
    return catalog_root_;
  }
  [[nodiscard]] const Sha256Digest& graph_snapshot_root() const noexcept {
    return graph_snapshot_root_;
  }
  [[nodiscard]] RuntimeProfileModel model() const noexcept { return model_; }
  [[nodiscard]] RuntimeProfileWeightFormat weight_format() const noexcept {
    return weight_format_;
  }
  [[nodiscard]] RuntimeProfileGpuFamily gpu_family() const noexcept {
    return gpu_family_;
  }
  [[nodiscard]] bool dspark_enabled() const noexcept {
    return dspark_enabled_;
  }
  [[nodiscard]] RuntimeProfileResidency residency() const noexcept {
    return residency_;
  }
  [[nodiscard]] std::span<const std::int32_t> device_ordinals() const noexcept {
    return device_ordinals_;
  }
  [[nodiscard]] const Sha256Digest& device_observation_root() const noexcept {
    return device_observation_root_;
  }
  [[nodiscard]] const RuntimeProfileRoots& reference_roots() const noexcept {
    return reference_roots_;
  }
  [[nodiscard]] std::uint64_t capacity_template_exact_bytes() const noexcept {
    return capacity_template_exact_bytes_;
  }
  [[nodiscard]] bool evidence_projection_bound() const noexcept {
    return evidence_projection_bound_;
  }
  [[nodiscard]] const RuntimeEvidenceRoots& evidence_roots() const noexcept {
    return evidence_roots_;
  }
  [[nodiscard]] RuntimeProfileReferenceRetention reference_retention()
      const noexcept { return reference_retention_; }
  [[nodiscard]] bool references_retained() const noexcept {
    return reference_retention_ != RuntimeProfileReferenceRetention::kNone &&
           reference_anchor_ != nullptr;
  }
  [[nodiscard]] bool production_references_leased() const noexcept {
    return reference_retention_ ==
               RuntimeProfileReferenceRetention::kImmutableDescriptorLease &&
           reference_anchor_ != nullptr;
  }
  [[nodiscard]] bool production_authority_leased() const noexcept {
    return authority_anchor_ != nullptr;
  }
  [[nodiscard]] bool production_ready() const noexcept {
    return production_references_leased() && production_authority_leased();
  }
  [[nodiscard]] bool production_compute_ready() const noexcept {
    return production_ready() && evidence_projection_bound_;
  }

 private:
  friend Result<RuntimeEngineAdmission> admit_runtime_engine(
      const VerifiedRuntimeProfile&, const VerifiedRuntimeProfileDevices&,
      RuntimeProfileModel, RuntimeProfileWeightFormat);
  friend Result<RuntimeEngineAdmission> retain_runtime_engine_reference_objects(
      RuntimeEngineAdmission, const RuntimeProfileReferenceClosure&,
      std::vector<std::vector<std::byte>>);
  friend Result<RuntimeEngineAdmission> retain_runtime_engine_reference_leases(
      RuntimeEngineAdmission, VerifiedRuntimeProfileReferenceLeases);
  friend Result<RuntimeEngineAdmission> retain_runtime_engine_authority_leases(
      RuntimeEngineAdmission, VerifiedRuntimeProfileAuthorityLeases);
  friend Result<RuntimeEngineAdmission> bind_runtime_engine_evidence_projection(
      RuntimeEngineAdmission, const RuntimeEvidenceProjection&);
  RuntimeEngineAdmission(std::string profile_id, std::string revision,
                         Sha256Digest policy_digest,
                         Sha256Digest catalog_root,
                         Sha256Digest envelope_root,
                         Sha256Digest reference_closure_root,
                         RuntimeProfileRoots reference_roots,
                         RuntimeProfileModel model,
                         RuntimeProfileWeightFormat weight_format,
                         RuntimeProfileGpuFamily gpu_family,
                         bool dspark_enabled,
                         RuntimeProfileResidency residency,
                         std::vector<std::int32_t> device_ordinals,
                         Sha256Digest device_observation_root) noexcept;
  std::string profile_id_;
  std::string revision_;
  Sha256Digest policy_digest_{};
  Sha256Digest catalog_root_{};
  Sha256Digest envelope_root_{};
  Sha256Digest reference_closure_root_{};
  RuntimeProfileRoots reference_roots_{};
  std::uint64_t capacity_template_exact_bytes_ = 0;
  RuntimeProfileModel model_ = RuntimeProfileModel::kQwen3_0_6B;
  RuntimeProfileWeightFormat weight_format_ = RuntimeProfileWeightFormat::kBf16;
  RuntimeProfileGpuFamily gpu_family_ =
      RuntimeProfileGpuFamily::kRtx4090D24GiB;
  bool dspark_enabled_ = false;
  RuntimeProfileResidency residency_ = RuntimeProfileResidency::kFullResident;
  std::vector<std::int32_t> device_ordinals_;
  Sha256Digest device_observation_root_{};
  RuntimeEvidenceRoots evidence_roots_{};
  bool evidence_projection_bound_ = false;
  Sha256Digest graph_snapshot_root_{};
  RuntimeProfileReferenceRetention reference_retention_ =
      RuntimeProfileReferenceRetention::kNone;
  std::shared_ptr<const void> reference_anchor_;
  std::shared_ptr<const void> authority_anchor_;
};

Result<RuntimeEngineAdmission> admit_runtime_engine(
    const VerifiedRuntimeProfile& profile,
    const VerifiedRuntimeProfileDevices& devices,
    RuntimeProfileModel expected_model,
    RuntimeProfileWeightFormat expected_weight_format);

}  // namespace pih
