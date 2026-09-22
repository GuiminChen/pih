#include "pih/model/deepseek_dspark_runtime_evidence.h"

#include <algorithm>
#include <array>
#include <utility>

namespace pih {
namespace {

constexpr std::array<std::byte, 8> kMagic{
    std::byte{'X'}, std::byte{'I'}, std::byte{'D'}, std::byte{'S'},
    std::byte{'P'}, std::byte{'R'}, std::byte{'M'}, std::byte{'1'}};

bool nonzero(const Sha256Digest& value) {
  return std::ranges::any_of(
      value.bytes, [](std::byte byte) { return byte != std::byte{0}; });
}

void append_root(std::vector<std::byte>& output, const Sha256Digest& root) {
  output.insert(output.end(), root.bytes.begin(), root.bytes.end());
}

Sha256Digest read_root(std::span<const std::byte> bytes,
                       std::size_t& cursor) {
  Sha256Digest result{};
  std::ranges::copy(bytes.subspan(cursor, result.bytes.size()),
                    result.bytes.begin());
  cursor += result.bytes.size();
  return result;
}

bool resolved_inputs_match(const DeepSeekDsparkResolvedInputRoots& inputs,
                           const RuntimeProfileRoots& roots) {
  return inputs.runtime_semantic_root == roots.runtime_semantic_root &&
         inputs.capacity_template_root == roots.capacity_template_root &&
         inputs.feature_selection_root == roots.feature_selection_root &&
         inputs.hardware_identity_root == roots.hardware_identity_root &&
         inputs.kernel_closure_root == roots.kernel_closure_root &&
         inputs.security_runtime_root == roots.security_runtime_root;
}

Status validate_admission_shape(const RuntimeEngineAdmission& admission) {
  if (!admission.production_ready()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark permit requires immutable production leases");
  }
  if (!admission.evidence_projection_bound()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark permit requires a verified evidence projection");
  }
  if (!nonzero(admission.graph_snapshot_root()) ||
      !nonzero(admission.device_observation_root())) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark permit requires bound graph and device identity");
  }
  if (admission.model() != RuntimeProfileModel::kDeepSeekV4Flash0731 ||
      admission.weight_format() !=
          RuntimeProfileWeightFormat::kDeepSeekNative ||
      admission.gpu_family() != RuntimeProfileGpuFamily::kH100Pcie80GiB ||
      !admission.dspark_enabled() || admission.device_ordinals().empty() ||
      admission.device_ordinals().size() > 4) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark permit is restricted to exact H100 profiles");
  }
  return Status::Ok();
}

}  // namespace

DeepSeekDsparkRuntimeEvidenceManifest::
DeepSeekDsparkRuntimeEvidenceManifest(
    RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
    RuntimeProfileResidency residency,
    DeepSeekDsparkResolvedInputRoots resolved_inputs,
    Sha256Digest artifact_root,
    DeepSeekDsparkQualificationRoots qualification_roots,
    std::vector<std::byte> canonical_bytes,
    Sha256Digest domain_decision_roots_digest) noexcept
    : gpu_family_(gpu_family), world_size_(world_size), residency_(residency),
      resolved_inputs_(resolved_inputs), artifact_root_(artifact_root),
      qualification_roots_(qualification_roots),
      canonical_bytes_(std::move(canonical_bytes)),
      domain_decision_roots_digest_(domain_decision_roots_digest) {}

Result<DeepSeekDsparkRuntimeEvidenceManifest>
DeepSeekDsparkRuntimeEvidenceManifest::Create(
    RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
    RuntimeProfileResidency residency,
    DeepSeekDsparkResolvedInputRoots resolved_inputs,
    Sha256Digest artifact_root,
    DeepSeekDsparkQualificationRoots qualification_roots) {
  const std::array<const Sha256Digest*, 11> roots{
      &artifact_root,
      &resolved_inputs.runtime_semantic_root,
      &resolved_inputs.capacity_template_root,
      &resolved_inputs.feature_selection_root,
      &resolved_inputs.hardware_identity_root,
      &resolved_inputs.kernel_closure_root,
      &resolved_inputs.security_runtime_root,
      &qualification_roots.official_generation_receipt_root,
      &qualification_roots.official_checkpoint_numerical_receipt_root,
      &qualification_roots.three_stage_runtime_numerical_receipt_root,
      &qualification_roots.exact_profile_hardware_receipt_root};
  if (gpu_family != RuntimeProfileGpuFamily::kH100Pcie80GiB ||
      world_size == 0 || world_size > 4 ||
      (residency != RuntimeProfileResidency::kHostSpill &&
       residency != RuntimeProfileResidency::kFullResident) ||
      std::ranges::any_of(roots,
                          [](const Sha256Digest* root) { return !nonzero(*root); })) {
    return Status::InvalidArgument(
        "DeepSeek DSpark runtime evidence manifest is invalid");
  }
  const std::array qualification{
      qualification_roots.official_generation_receipt_root,
      qualification_roots.official_checkpoint_numerical_receipt_root,
      qualification_roots.three_stage_runtime_numerical_receipt_root,
      qualification_roots.exact_profile_hardware_receipt_root};
  for (std::size_t left = 0; left < qualification.size(); ++left) {
    for (std::size_t right = left + 1; right < qualification.size(); ++right) {
      if (qualification[left] == qualification[right]) {
        return Status::InvalidArgument(
            "DeepSeek DSpark qualification roles share one receipt root");
      }
    }
  }

  std::vector<std::byte> bytes(kMagic.begin(), kMagic.end());
  bytes.push_back(std::byte{1});
  bytes.push_back(static_cast<std::byte>(gpu_family));
  bytes.push_back(static_cast<std::byte>(world_size));
  bytes.push_back(static_cast<std::byte>(residency));
  for (const auto* root : roots) append_root(bytes, *root);
  if (bytes.size() != kCanonicalBytes) {
    return Status::Internal(
        "DeepSeek DSpark runtime evidence encoding size drifted");
  }
  auto digest = sha256(bytes);
  if (!digest.ok()) return digest.status();
  return DeepSeekDsparkRuntimeEvidenceManifest(
      gpu_family, world_size, residency, resolved_inputs, artifact_root,
      qualification_roots, std::move(bytes), *digest);
}

Result<DeepSeekDsparkRuntimeEvidenceManifest>
parse_deepseek_dspark_runtime_evidence_manifest(
    std::span<const std::byte> bytes) {
  if (bytes.size() != DeepSeekDsparkRuntimeEvidenceManifest::kCanonicalBytes ||
      !std::ranges::equal(bytes.first(kMagic.size()), kMagic) ||
      bytes[8] != std::byte{1}) {
    return Status::InvalidArgument(
        "DeepSeek DSpark runtime evidence encoding is invalid");
  }
  const auto gpu_family = static_cast<RuntimeProfileGpuFamily>(bytes[9]);
  const auto world_size = static_cast<std::uint8_t>(bytes[10]);
  const auto residency = static_cast<RuntimeProfileResidency>(bytes[11]);
  std::size_t cursor = 12;
  const auto artifact_root = read_root(bytes, cursor);
  DeepSeekDsparkResolvedInputRoots resolved{};
  resolved.runtime_semantic_root = read_root(bytes, cursor);
  resolved.capacity_template_root = read_root(bytes, cursor);
  resolved.feature_selection_root = read_root(bytes, cursor);
  resolved.hardware_identity_root = read_root(bytes, cursor);
  resolved.kernel_closure_root = read_root(bytes, cursor);
  resolved.security_runtime_root = read_root(bytes, cursor);
  DeepSeekDsparkQualificationRoots qualification{};
  qualification.official_generation_receipt_root = read_root(bytes, cursor);
  qualification.official_checkpoint_numerical_receipt_root =
      read_root(bytes, cursor);
  qualification.three_stage_runtime_numerical_receipt_root =
      read_root(bytes, cursor);
  qualification.exact_profile_hardware_receipt_root = read_root(bytes, cursor);
  auto result = DeepSeekDsparkRuntimeEvidenceManifest::Create(
      gpu_family, world_size, residency, resolved, artifact_root,
      qualification);
  if (!result.ok() ||
      !std::ranges::equal(result->canonical_bytes(), bytes)) {
    return Status::InvalidArgument(
        "DeepSeek DSpark runtime evidence encoding is noncanonical");
  }
  return result;
}

DeepSeekDsparkRuntimePermit::DeepSeekDsparkRuntimePermit(
    Sha256Digest envelope_root, Sha256Digest reference_closure_root,
    Sha256Digest graph_snapshot_root, Sha256Digest device_observation_root,
    Sha256Digest artifact_root, Sha256Digest evidence_bundle_root,
    Sha256Digest domain_decision_roots_digest,
    RuntimeProfileResidency residency,
    std::vector<std::int32_t> device_ordinals) noexcept
    : envelope_root_(envelope_root),
      reference_closure_root_(reference_closure_root),
      graph_snapshot_root_(graph_snapshot_root),
      device_observation_root_(device_observation_root),
      artifact_root_(artifact_root),
      evidence_bundle_root_(evidence_bundle_root),
      domain_decision_roots_digest_(domain_decision_roots_digest),
      world_size_(static_cast<std::uint8_t>(device_ordinals.size())),
      residency_(residency), device_ordinals_(std::move(device_ordinals)) {}

Result<DeepSeekDsparkRuntimePermit> issue_deepseek_dspark_runtime_permit(
    const RuntimeEngineAdmission& admission,
    const DeepSeekDsparkRuntimeEvidenceManifest& evidence,
    const Sha256Digest& artifact_root) {
  const auto shape = validate_admission_shape(admission);
  if (!shape.ok()) return shape;
  if (!(artifact_root == evidence.artifact_root()) ||
      evidence.gpu_family() != admission.gpu_family() ||
      evidence.world_size() != admission.device_ordinals().size() ||
      evidence.residency() != admission.residency() ||
      !resolved_inputs_match(evidence.resolved_inputs(),
                             admission.reference_roots()) ||
      !(evidence.domain_decision_roots_digest() ==
        admission.evidence_roots().domain_decision_roots_digest)) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark evidence differs from the admitted exact profile");
  }
  return DeepSeekDsparkRuntimePermit(
      admission.envelope_root(), admission.reference_closure_root(),
      admission.graph_snapshot_root(), admission.device_observation_root(),
      artifact_root, admission.evidence_roots().bundle_root,
      admission.evidence_roots().domain_decision_roots_digest,
      admission.residency(),
      std::vector<std::int32_t>(admission.device_ordinals().begin(),
                                admission.device_ordinals().end()));
}

Status validate_deepseek_dspark_runtime_permit(
    const DeepSeekDsparkRuntimePermit& permit,
    const RuntimeEngineAdmission& admission,
    const Sha256Digest& artifact_root) {
  const auto shape = validate_admission_shape(admission);
  if (!shape.ok()) return shape;
  if (!(permit.envelope_root_ == admission.envelope_root()) ||
      !(permit.reference_closure_root_ == admission.reference_closure_root()) ||
      !(permit.graph_snapshot_root_ == admission.graph_snapshot_root()) ||
      !(permit.device_observation_root_ ==
        admission.device_observation_root()) ||
      !(permit.artifact_root_ == artifact_root) ||
      !(permit.evidence_bundle_root_ == admission.evidence_roots().bundle_root) ||
      !(permit.domain_decision_roots_digest_ ==
        admission.evidence_roots().domain_decision_roots_digest) ||
      permit.world_size_ != admission.device_ordinals().size() ||
      permit.residency_ != admission.residency() ||
      !std::ranges::equal(permit.device_ordinals_,
                          admission.device_ordinals())) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark runtime permit was spliced across admission axes");
  }
  return Status::Ok();
}

}  // namespace pih
