#include "pih/model/runtime_profile_payload.h"

#include <algorithm>

#include "pih/model/runtime_profile_device_gate.h"

namespace pih {
namespace {

constexpr std::string_view kDomain = "pih:runtime-profile-payload:v1";

bool nonzero(const Sha256Digest& value) {
  return std::any_of(value.bytes.begin(), value.bytes.end(),
                     [](std::byte byte) { return byte != std::byte{0}; });
}

bool valid_roots(const RuntimeProfileRoots& roots) {
  return nonzero(roots.runtime_semantic_root) &&
         nonzero(roots.capacity_template_root) &&
         nonzero(roots.feature_selection_root) &&
         nonzero(roots.release_evidence_root) &&
         nonzero(roots.hardware_identity_root) &&
         nonzero(roots.kernel_closure_root) &&
         nonzero(roots.security_runtime_root);
}

void append_root(std::vector<std::byte>& output, const Sha256Digest& root) {
  output.insert(output.end(), root.bytes.begin(), root.bytes.end());
}

bool valid_axes(RuntimeProfileModel model,
                RuntimeProfileWeightFormat weight_format,
                RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
                bool dspark_enabled, RuntimeProfileResidency residency,
                RuntimeProfileEvidenceState evidence_state) {
  if (world_size < 1 || world_size > 4 ||
      (evidence_state != RuntimeProfileEvidenceState::kHardwareEvidenceOpen &&
       evidence_state != RuntimeProfileEvidenceState::kCorrectnessSupported)) {
    return false;
  }
  if (model == RuntimeProfileModel::kQwen3_0_6B) {
    return (weight_format == RuntimeProfileWeightFormat::kBf16 ||
            weight_format == RuntimeProfileWeightFormat::kXingInt4) &&
           (gpu_family == RuntimeProfileGpuFamily::kRtx4090D24GiB ||
            gpu_family == RuntimeProfileGpuFamily::kH100Pcie80GiB) &&
           world_size == 1 && !dspark_enabled &&
           residency == RuntimeProfileResidency::kFullResident;
  }
  if (model != RuntimeProfileModel::kDeepSeekV4Flash0731 ||
      weight_format != RuntimeProfileWeightFormat::kDeepSeekNative ||
      (gpu_family != RuntimeProfileGpuFamily::kRtx4090D24GiB &&
       gpu_family != RuntimeProfileGpuFamily::kH100Pcie80GiB)) {
    return false;
  }
  if (gpu_family == RuntimeProfileGpuFamily::kRtx4090D24GiB) {
    return !dspark_enabled && residency == RuntimeProfileResidency::kHostSpill;
  }
  if (world_size <= 2 && residency != RuntimeProfileResidency::kHostSpill) {
    return false;
  }
  return true;
}

Status invalid() {
  return Status::InvalidArgument("runtime profile payload is invalid");
}

}  // namespace

RuntimeProfilePayload::RuntimeProfilePayload(
    RuntimeProfileModel model, RuntimeProfileWeightFormat weight_format,
    RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
    bool dspark_enabled, RuntimeProfileResidency residency,
    RuntimeProfileEvidenceState evidence_state, RuntimeProfileRoots roots,
    std::vector<std::byte> canonical_bytes) noexcept
    : model_(model), weight_format_(weight_format), gpu_family_(gpu_family),
      world_size_(world_size), dspark_enabled_(dspark_enabled),
      residency_(residency), evidence_state_(evidence_state), roots_(roots),
      canonical_bytes_(std::move(canonical_bytes)) {}

Result<RuntimeProfilePayload> RuntimeProfilePayload::Create(
    RuntimeProfileModel model, RuntimeProfileWeightFormat weight_format,
    RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
    bool dspark_enabled, RuntimeProfileResidency residency,
    RuntimeProfileEvidenceState evidence_state, RuntimeProfileRoots roots) {
  if (!valid_axes(model, weight_format, gpu_family, world_size, dspark_enabled,
                  residency, evidence_state) ||
      !valid_roots(roots)) {
    return invalid();
  }
  std::vector<std::byte> canonical;
  const auto domain = std::as_bytes(std::span(kDomain));
  canonical.insert(canonical.end(), domain.begin(), domain.end());
  canonical.push_back(std::byte{0});
  canonical.push_back(static_cast<std::byte>(model));
  canonical.push_back(static_cast<std::byte>(weight_format));
  canonical.push_back(static_cast<std::byte>(gpu_family));
  canonical.push_back(static_cast<std::byte>(world_size));
  canonical.push_back(dspark_enabled ? std::byte{1} : std::byte{0});
  canonical.push_back(static_cast<std::byte>(residency));
  canonical.push_back(static_cast<std::byte>(evidence_state));
  append_root(canonical, roots.runtime_semantic_root);
  append_root(canonical, roots.capacity_template_root);
  append_root(canonical, roots.feature_selection_root);
  append_root(canonical, roots.release_evidence_root);
  append_root(canonical, roots.hardware_identity_root);
  append_root(canonical, roots.kernel_closure_root);
  append_root(canonical, roots.security_runtime_root);
  return RuntimeProfilePayload(model, weight_format, gpu_family, world_size,
                               dspark_enabled, residency, evidence_state, roots,
                               std::move(canonical));
}

Result<RuntimeProfilePayload> parse_runtime_profile_payload(
    std::span<const std::byte> bytes) {
  const std::size_t prefix = kDomain.size() + 1;
  const std::size_t expected = prefix + 7 + 7 * 32;
  if (bytes.size() != expected) return invalid();
  const auto domain = std::as_bytes(std::span(kDomain));
  if (!std::equal(domain.begin(), domain.end(), bytes.begin()) ||
      bytes[kDomain.size()] != std::byte{0}) {
    return invalid();
  }
  const auto value = [&](std::size_t index) {
    return std::to_integer<std::uint8_t>(bytes[prefix + index]);
  };
  if (value(4) > 1) return invalid();
  RuntimeProfileRoots roots{};
  std::array<Sha256Digest*, 7> targets{
      &roots.runtime_semantic_root, &roots.capacity_template_root,
      &roots.feature_selection_root, &roots.release_evidence_root,
      &roots.hardware_identity_root, &roots.kernel_closure_root,
      &roots.security_runtime_root};
  auto cursor = bytes.begin() + static_cast<std::ptrdiff_t>(prefix + 7);
  for (auto* target : targets) {
    std::copy_n(cursor, 32, target->bytes.begin());
    cursor += 32;
  }
  auto result = RuntimeProfilePayload::Create(
      static_cast<RuntimeProfileModel>(value(0)),
      static_cast<RuntimeProfileWeightFormat>(value(1)),
      static_cast<RuntimeProfileGpuFamily>(value(2)), value(3), value(4) == 1,
      static_cast<RuntimeProfileResidency>(value(5)),
      static_cast<RuntimeProfileEvidenceState>(value(6)), roots);
  if (!result.ok() || !std::equal(result->canonical_bytes().begin(),
                                  result->canonical_bytes().end(),
                                  bytes.begin())) {
    return invalid();
  }
  return result;
}

VerifiedRuntimeProfile::VerifiedRuntimeProfile(
    std::string profile_id, std::string revision, Sha256Digest policy_digest,
    Sha256Digest catalog_root, Sha256Digest envelope_root,
    Sha256Digest reference_closure_root,
    RuntimeProfilePayload payload) noexcept
    : profile_id_(std::move(profile_id)), revision_(std::move(revision)),
      policy_digest_(policy_digest), catalog_root_(catalog_root),
      envelope_root_(envelope_root),
      reference_closure_root_(reference_closure_root),
      payload_(std::move(payload)) {}

Result<VerifiedRuntimeProfile> bind_verified_runtime_profile(
    const VerifiedProfileAuthority& authority,
    const SignedProfileEnvelope& envelope) {
  if (authority.profile_id() != envelope.profile_id() ||
      authority.revision() != envelope.revision() ||
      !(authority.envelope_root() == envelope.envelope_root()) ||
      !(authority.payload_root() == envelope.payload_root()) ||
      !(authority.reference_closure_root() ==
        envelope.reference_closure_root())) {
    return Status::FailedPrecondition(
        "verified authority does not bind this profile envelope");
  }
  auto payload = parse_runtime_profile_payload(envelope.payload_bytes());
  if (!payload.ok()) return payload.status();
  return VerifiedRuntimeProfile(std::string(authority.profile_id()),
                                std::string(authority.revision()),
                                authority.policy_digest(),
                                authority.catalog_root(),
                                authority.envelope_root(),
                                authority.reference_closure_root(),
                                std::move(*payload));
}

RuntimeEngineAdmission::RuntimeEngineAdmission(
    std::string profile_id, std::string revision, Sha256Digest policy_digest,
    Sha256Digest catalog_root,
    Sha256Digest envelope_root, Sha256Digest reference_closure_root,
    RuntimeProfileRoots reference_roots, RuntimeProfileModel model,
    RuntimeProfileWeightFormat weight_format,
    RuntimeProfileGpuFamily gpu_family,
    bool dspark_enabled, RuntimeProfileResidency residency,
    std::vector<std::int32_t> device_ordinals,
    Sha256Digest device_observation_root) noexcept
    : profile_id_(std::move(profile_id)), revision_(std::move(revision)),
      policy_digest_(policy_digest), catalog_root_(catalog_root),
      envelope_root_(envelope_root),
      reference_closure_root_(reference_closure_root),
      reference_roots_(reference_roots), model_(model), weight_format_(weight_format),
      gpu_family_(gpu_family), dspark_enabled_(dspark_enabled),
      residency_(residency), device_ordinals_(std::move(device_ordinals)),
      device_observation_root_(device_observation_root) {}

Result<RuntimeEngineAdmission> admit_runtime_engine(
    const VerifiedRuntimeProfile& profile,
    const VerifiedRuntimeProfileDevices& devices,
    RuntimeProfileModel expected_model,
    RuntimeProfileWeightFormat expected_weight_format) {
  const auto& payload = profile.payload();
  if (payload.evidence_state() !=
      RuntimeProfileEvidenceState::kCorrectnessSupported) {
    return Status::FailedPrecondition(
        "runtime profile hardware evidence remains open");
  }
  if (payload.model() != expected_model ||
      payload.weight_format() != expected_weight_format ||
      payload.gpu_family() != devices.gpu_family() ||
      payload.world_size() != devices.ordinals().size()) {
    return Status::FailedPrecondition(
        "runtime profile does not authorize the requested engine");
  }
  return RuntimeEngineAdmission(std::string(profile.profile_id()),
                                std::string(profile.revision()),
                                profile.policy_digest(), profile.catalog_root(),
                                profile.envelope_root(),
                                profile.reference_closure_root(), payload.roots(),
                                payload.model(),
                                payload.weight_format(), payload.gpu_family(),
                                payload.dspark_enabled(), payload.residency(),
                                std::vector<std::int32_t>(
                                    devices.ordinals().begin(),
                                    devices.ordinals().end()),
                                devices.observation_root());
}

}  // namespace pih
