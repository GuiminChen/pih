#include "pih/model/runtime_evidence_projection.h"

#include <algorithm>
#include <array>

namespace pih {
namespace {

constexpr std::array<std::byte, 8> kMagic{
    std::byte{'X'}, std::byte{'I'}, std::byte{'E'}, std::byte{'V'},
    std::byte{'P'}, std::byte{'R'}, std::byte{'1'}, std::byte{0}};
constexpr std::size_t kCanonicalBytes = 8 + 1 + 3 + 5 * 32;

bool nonzero(const Sha256Digest& value) {
  return std::ranges::any_of(value.bytes,
      [](std::byte value) { return value != std::byte{0}; });
}

bool valid(RuntimeEvidenceBundleState value) {
  return value >= RuntimeEvidenceBundleState::kAssembling &&
         value <= RuntimeEvidenceBundleState::kRevoked;
}
bool valid(RuntimeEvidenceReplayState value) {
  return value >= RuntimeEvidenceReplayState::kNotRun &&
         value <= RuntimeEvidenceReplayState::kFailed;
}
bool valid(RuntimeHardwareQualificationState value) {
  return value >= RuntimeHardwareQualificationState::kSupported &&
         value <= RuntimeHardwareQualificationState::kRejected;
}

}  // namespace

RuntimeEvidenceProjection::RuntimeEvidenceProjection(
    RuntimeEvidenceBundleState bundle_state,
    RuntimeEvidenceReplayState replay_state,
    RuntimeHardwareQualificationState hardware_state,
    RuntimeEvidenceRoots roots, std::vector<std::byte> canonical_bytes,
    Sha256Digest projection_root) noexcept
    : bundle_state_(bundle_state), replay_state_(replay_state),
      hardware_state_(hardware_state), roots_(roots),
      canonical_bytes_(std::move(canonical_bytes)),
      projection_root_(projection_root) {}

Result<RuntimeEvidenceProjection> RuntimeEvidenceProjection::Create(
    RuntimeEvidenceBundleState bundle_state,
    RuntimeEvidenceReplayState replay_state,
    RuntimeHardwareQualificationState hardware_state,
    RuntimeEvidenceRoots roots) {
  if (!valid(bundle_state) || !valid(replay_state) || !valid(hardware_state) ||
      !nonzero(roots.bundle_root) || !nonzero(roots.audit_closure_root) ||
      !nonzero(roots.required_role_schema_root) ||
      !nonzero(roots.domain_decision_roots_digest) ||
      !nonzero(roots.builder_policy_root)) {
    return Status::InvalidArgument("runtime evidence projection is invalid");
  }
  std::vector<std::byte> bytes(kMagic.begin(), kMagic.end());
  bytes.push_back(std::byte{1});
  bytes.push_back(static_cast<std::byte>(bundle_state));
  bytes.push_back(static_cast<std::byte>(replay_state));
  bytes.push_back(static_cast<std::byte>(hardware_state));
  for (const auto* root : {&roots.bundle_root, &roots.audit_closure_root,
                           &roots.required_role_schema_root,
                           &roots.domain_decision_roots_digest,
                           &roots.builder_policy_root}) {
    bytes.insert(bytes.end(), root->bytes.begin(), root->bytes.end());
  }
  auto digest = sha256(bytes);
  if (!digest.ok()) return digest.status();
  return RuntimeEvidenceProjection(bundle_state, replay_state, hardware_state,
                                   roots, std::move(bytes), *digest);
}

Result<RuntimeEvidenceProjection> parse_runtime_evidence_projection(
    std::span<const std::byte> bytes) {
  if (bytes.size() != kCanonicalBytes ||
      !std::ranges::equal(bytes.first(kMagic.size()), kMagic) ||
      bytes[8] != std::byte{1}) {
    return Status::InvalidArgument("runtime evidence projection encoding is invalid");
  }
  const auto bundle = static_cast<RuntimeEvidenceBundleState>(bytes[9]);
  const auto replay = static_cast<RuntimeEvidenceReplayState>(bytes[10]);
  const auto hardware = static_cast<RuntimeHardwareQualificationState>(bytes[11]);
  RuntimeEvidenceRoots roots{};
  std::size_t cursor = 12;
  for (auto* root : {&roots.bundle_root, &roots.audit_closure_root,
                     &roots.required_role_schema_root,
                     &roots.domain_decision_roots_digest,
                     &roots.builder_policy_root}) {
    std::ranges::copy(bytes.subspan(cursor, root->bytes.size()),
                      root->bytes.begin());
    cursor += root->bytes.size();
  }
  auto result = RuntimeEvidenceProjection::Create(bundle, replay, hardware, roots);
  if (!result.ok() || !std::ranges::equal(result->canonical_bytes(), bytes)) {
    return Status::InvalidArgument("runtime evidence projection is noncanonical");
  }
  return result;
}

Status verify_runtime_evidence_projection(
    const VerifiedRuntimeProfile& profile,
    const RuntimeEvidenceProjection& projection) {
  if (!(projection.projection_root() ==
        profile.payload().roots().release_evidence_root)) {
    return Status::FailedPrecondition(
        "runtime evidence projection differs from profile root");
  }
  if (profile.payload().evidence_state() ==
      RuntimeProfileEvidenceState::kCorrectnessSupported) {
    if (projection.bundle_state() != RuntimeEvidenceBundleState::kComplete ||
        projection.replay_state() != RuntimeEvidenceReplayState::kPassed ||
        projection.hardware_state() !=
            RuntimeHardwareQualificationState::kSupported) {
      return Status::FailedPrecondition(
          "supported profile lacks complete replayed hardware evidence");
    }
  }
  return Status::Ok();
}

Result<RuntimeEngineAdmission> bind_runtime_engine_evidence_projection(
    RuntimeEngineAdmission admission,
    const RuntimeEvidenceProjection& projection) {
  if (!(projection.projection_root() ==
        admission.reference_roots_.release_evidence_root) ||
      projection.bundle_state() != RuntimeEvidenceBundleState::kComplete ||
      projection.replay_state() != RuntimeEvidenceReplayState::kPassed ||
      projection.hardware_state() !=
          RuntimeHardwareQualificationState::kSupported) {
    return Status::FailedPrecondition(
        "runtime evidence projection does not authorize this admission");
  }
  admission.evidence_roots_ = projection.roots();
  admission.evidence_projection_bound_ = true;
  return admission;
}

}  // namespace pih
