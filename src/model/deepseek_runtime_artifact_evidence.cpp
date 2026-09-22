#include "pih/model/deepseek_runtime_artifact_evidence.h"

#include <algorithm>
#include <array>
#include <utility>

#include "pih/core/canonical_hash.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_runtime_evidence.h"
#endif

namespace pih {
namespace {

constexpr std::array<std::byte, 8> kMagic{
    std::byte{'X'}, std::byte{'I'}, std::byte{'D'}, std::byte{'A'},
    std::byte{'R'}, std::byte{'T'}, std::byte{'M'}, std::byte{'1'}};

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

bool valid_main_path_axes(RuntimeProfileGpuFamily family,
                          std::uint8_t world_size,
                          RuntimeProfileResidency residency) noexcept {
  if (world_size < 1 || world_size > 4) return false;
  if (family == RuntimeProfileGpuFamily::kRtx4090D24GiB) {
    return residency == RuntimeProfileResidency::kHostSpill;
  }
  if (family != RuntimeProfileGpuFamily::kH100Pcie80GiB) return false;
  return world_size <= 2 ? residency == RuntimeProfileResidency::kHostSpill
                         : (residency == RuntimeProfileResidency::kHostSpill ||
                            residency == RuntimeProfileResidency::kFullResident);
}

std::array<const Sha256Digest*, 10> manifest_roots(
    const DeepSeekRuntimeResolvedInputRoots& inputs,
    const Sha256Digest& artifact,
    const DeepSeekRuntimeArtifactQualificationRoots& qualification) {
  return {&artifact,
          &inputs.runtime_semantic_root,
          &inputs.capacity_template_root,
          &inputs.feature_selection_root,
          &inputs.hardware_identity_root,
          &inputs.kernel_closure_root,
          &inputs.security_runtime_root,
          &qualification.official_generation_receipt_root,
          &qualification.official_checkpoint_numerical_receipt_root,
          &qualification.exact_profile_hardware_receipt_root};
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

bool resolved_inputs_match(const DeepSeekRuntimeResolvedInputRoots& inputs,
                           const RuntimeProfileRoots& roots) noexcept {
  return inputs.runtime_semantic_root == roots.runtime_semantic_root &&
         inputs.capacity_template_root == roots.capacity_template_root &&
         inputs.feature_selection_root == roots.feature_selection_root &&
         inputs.hardware_identity_root == roots.hardware_identity_root &&
         inputs.kernel_closure_root == roots.kernel_closure_root &&
         inputs.security_runtime_root == roots.security_runtime_root;
}

Result<Sha256Digest> compile_binding_root(
    const RuntimeEngineAdmission& admission,
    const Sha256Digest& artifact_root,
    const Sha256Digest& evidence_manifest_root) {
  if (!admission.production_ready() ||
      !admission.evidence_projection_bound() ||
      admission.model() != RuntimeProfileModel::kDeepSeekV4Flash0731 ||
      admission.weight_format() != RuntimeProfileWeightFormat::kDeepSeekNative ||
      admission.device_ordinals().empty() ||
      admission.device_ordinals().size() > 4 ||
      admission.profile_id().empty() || admission.revision().empty() ||
      !nonzero(admission.graph_snapshot_root()) ||
      !nonzero(admission.device_observation_root()) ||
      !nonzero(artifact_root) || !nonzero(evidence_manifest_root) ||
      evidence_manifest_root !=
          admission.evidence_roots().domain_decision_roots_digest) {
    return Status::FailedPrecondition(
        "DeepSeek artifact binding requires exact production admission evidence");
  }

  auto ordinal_builder = CanonicalHashBuilder::Create(
      "pih:deepseek-runtime-artifact-device-ordinals:v1",
      static_cast<std::uint32_t>(admission.device_ordinals().size() + 1U));
  if (!ordinal_builder.ok()) return ordinal_builder.status();
  auto status = ordinal_builder->add_u32(
      1, static_cast<std::uint32_t>(admission.device_ordinals().size()));
  for (std::size_t index = 0;
       status.ok() && index < admission.device_ordinals().size(); ++index) {
    if (admission.device_ordinals()[index] < 0) {
      return Status::FailedPrecondition(
          "DeepSeek artifact binding device ordinal is invalid");
    }
    status = ordinal_builder->add_u32(
        static_cast<std::uint16_t>(index + 2U),
        static_cast<std::uint32_t>(admission.device_ordinals()[index]));
  }
  if (!status.ok()) return status;
  auto ordinal_root = ordinal_builder->finalize();
  if (!ordinal_root.ok()) return ordinal_root.status();

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-runtime-artifact-admission-binding:v1", 29);
  if (!builder.ok()) return builder.status();
  status = builder->add_ascii_utf8(1, admission.profile_id());
  if (status.ok()) status = builder->add_ascii_utf8(2, admission.revision());
  if (status.ok()) status = builder->add_hash(3, admission.policy_digest());
  if (status.ok()) status = builder->add_hash(4, admission.catalog_root());
  if (status.ok()) status = builder->add_hash(5, admission.envelope_root());
  if (status.ok()) {
    status = builder->add_hash(6, admission.reference_closure_root());
  }
  if (status.ok()) {
    status = builder->add_hash(7, admission.graph_snapshot_root());
  }
  const auto& roots = admission.reference_roots();
  if (status.ok()) status = builder->add_hash(8, roots.runtime_semantic_root);
  if (status.ok()) status = builder->add_hash(9, roots.capacity_template_root);
  if (status.ok()) status = builder->add_hash(10, roots.feature_selection_root);
  if (status.ok()) status = builder->add_hash(11, roots.release_evidence_root);
  if (status.ok()) status = builder->add_hash(12, roots.hardware_identity_root);
  if (status.ok()) status = builder->add_hash(13, roots.kernel_closure_root);
  if (status.ok()) status = builder->add_hash(14, roots.security_runtime_root);
  const auto& evidence = admission.evidence_roots();
  if (status.ok()) status = builder->add_hash(15, evidence.bundle_root);
  if (status.ok()) status = builder->add_hash(16, evidence.audit_closure_root);
  if (status.ok()) {
    status = builder->add_hash(17, evidence.required_role_schema_root);
  }
  if (status.ok()) {
    status = builder->add_hash(18, evidence.domain_decision_roots_digest);
  }
  if (status.ok()) status = builder->add_hash(19, evidence.builder_policy_root);
  if (status.ok()) {
    status = builder->add_hash(20, admission.device_observation_root());
  }
  if (status.ok()) {
    status = builder->add_u32(21, static_cast<std::uint32_t>(admission.model()));
  }
  if (status.ok()) {
    status = builder->add_u32(
        22, static_cast<std::uint32_t>(admission.weight_format()));
  }
  if (status.ok()) {
    status = builder->add_u32(
        23, static_cast<std::uint32_t>(admission.gpu_family()));
  }
  if (status.ok()) {
    status = builder->add_u32(
        24, static_cast<std::uint32_t>(admission.device_ordinals().size()));
  }
  if (status.ok()) {
    status = builder->add_u32(25, admission.dspark_enabled() ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u32(
        26, static_cast<std::uint32_t>(admission.residency()));
  }
  if (status.ok()) status = builder->add_hash(27, *ordinal_root);
  if (status.ok()) status = builder->add_hash(28, artifact_root);
  if (status.ok()) status = builder->add_hash(29, evidence_manifest_root);
  if (!status.ok()) return status;
  return builder->finalize();
}

Status validate_manifest_admission(
    const RuntimeEngineAdmission& admission,
    const DeepSeekRuntimeArtifactEvidenceManifest& evidence) {
  if (admission.dspark_enabled() ||
      admission.gpu_family() != evidence.gpu_family() ||
      admission.device_ordinals().size() != evidence.world_size() ||
      admission.residency() != evidence.residency() ||
      !resolved_inputs_match(evidence.resolved_inputs(),
                             admission.reference_roots()) ||
      evidence.domain_decision_roots_digest() !=
          admission.evidence_roots().domain_decision_roots_digest) {
    return Status::FailedPrecondition(
        "DeepSeek runtime artifact evidence differs from exact admission");
  }
  return Status::Ok();
}

}  // namespace

DeepSeekRuntimeArtifactEvidenceManifest::
DeepSeekRuntimeArtifactEvidenceManifest(
    RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
    RuntimeProfileResidency residency,
    DeepSeekRuntimeResolvedInputRoots resolved_inputs,
    Sha256Digest artifact_root,
    DeepSeekRuntimeArtifactQualificationRoots qualification_roots,
    std::vector<std::byte> canonical_bytes,
    Sha256Digest domain_decision_roots_digest) noexcept
    : gpu_family_(gpu_family), world_size_(world_size), residency_(residency),
      resolved_inputs_(resolved_inputs), artifact_root_(artifact_root),
      qualification_roots_(qualification_roots),
      canonical_bytes_(std::move(canonical_bytes)),
      domain_decision_roots_digest_(domain_decision_roots_digest) {}

Result<DeepSeekRuntimeArtifactEvidenceManifest>
DeepSeekRuntimeArtifactEvidenceManifest::Create(
    RuntimeProfileGpuFamily gpu_family, std::uint8_t world_size,
    RuntimeProfileResidency residency,
    DeepSeekRuntimeResolvedInputRoots resolved_inputs,
    Sha256Digest artifact_root,
    DeepSeekRuntimeArtifactQualificationRoots qualification_roots) {
  const auto roots = manifest_roots(
      resolved_inputs, artifact_root, qualification_roots);
  const std::array qualification{
      qualification_roots.official_generation_receipt_root,
      qualification_roots.official_checkpoint_numerical_receipt_root,
      qualification_roots.exact_profile_hardware_receipt_root};
  if (!valid_main_path_axes(gpu_family, world_size, residency) ||
      std::ranges::any_of(roots, [](const auto* root) {
        return !nonzero(*root);
      })) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact evidence manifest is invalid");
  }
  for (std::size_t left = 0; left < qualification.size(); ++left) {
    for (std::size_t right = left + 1; right < qualification.size(); ++right) {
      if (qualification[left] == qualification[right]) {
        return Status::InvalidArgument(
            "DeepSeek artifact qualification roles share one receipt root");
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
        "DeepSeek runtime artifact evidence encoding size drifted");
  }
  auto digest = sha256(bytes);
  if (!digest.ok()) return digest.status();
  return DeepSeekRuntimeArtifactEvidenceManifest(
      gpu_family, world_size, residency, resolved_inputs, artifact_root,
      qualification_roots, std::move(bytes), *digest);
}

Result<DeepSeekRuntimeArtifactEvidenceManifest>
parse_deepseek_runtime_artifact_evidence_manifest(
    std::span<const std::byte> bytes) {
  if (bytes.size() != DeepSeekRuntimeArtifactEvidenceManifest::kCanonicalBytes ||
      !std::ranges::equal(bytes.first(kMagic.size()), kMagic) ||
      bytes[8] != std::byte{1}) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact evidence encoding is invalid");
  }
  const auto family = static_cast<RuntimeProfileGpuFamily>(bytes[9]);
  const auto world_size = std::to_integer<std::uint8_t>(bytes[10]);
  const auto residency = static_cast<RuntimeProfileResidency>(bytes[11]);
  std::size_t cursor = 12;
  const auto artifact_root = read_root(bytes, cursor);
  DeepSeekRuntimeResolvedInputRoots inputs{};
  inputs.runtime_semantic_root = read_root(bytes, cursor);
  inputs.capacity_template_root = read_root(bytes, cursor);
  inputs.feature_selection_root = read_root(bytes, cursor);
  inputs.hardware_identity_root = read_root(bytes, cursor);
  inputs.kernel_closure_root = read_root(bytes, cursor);
  inputs.security_runtime_root = read_root(bytes, cursor);
  DeepSeekRuntimeArtifactQualificationRoots qualification{};
  qualification.official_generation_receipt_root = read_root(bytes, cursor);
  qualification.official_checkpoint_numerical_receipt_root =
      read_root(bytes, cursor);
  qualification.exact_profile_hardware_receipt_root = read_root(bytes, cursor);
  auto result = DeepSeekRuntimeArtifactEvidenceManifest::Create(
      family, world_size, residency, inputs, artifact_root, qualification);
  if (!result.ok() || !std::ranges::equal(result->canonical_bytes(), bytes)) {
    return Status::InvalidArgument(
        "DeepSeek runtime artifact evidence encoding is noncanonical");
  }
  return result;
}

DeepSeekRuntimeArtifactAdmissionBinding::
DeepSeekRuntimeArtifactAdmissionBinding(
    Sha256Digest artifact_root, Sha256Digest evidence_manifest_root,
    Sha256Digest binding_root,
    std::shared_ptr<const RuntimeEngineAdmission> admission_anchor) noexcept
    : artifact_root_(artifact_root),
      evidence_manifest_root_(evidence_manifest_root),
      binding_root_(binding_root),
      admission_anchor_(std::move(admission_anchor)) {}

Result<DeepSeekRuntimeArtifactAdmissionBinding>
DeepSeekRuntimeArtifactAdmissionBinding::Issue(
    const RuntimeEngineAdmission& admission,
    const DeepSeekRuntimeArtifactEvidenceManifest& evidence) {
  auto status = validate_manifest_admission(admission, evidence);
  if (!status.ok()) return status;
  auto root = compile_binding_root(
      admission, evidence.artifact_root(),
      evidence.domain_decision_roots_digest());
  if (!root.ok()) return root.status();
  return DeepSeekRuntimeArtifactAdmissionBinding(
      evidence.artifact_root(), evidence.domain_decision_roots_digest(),
      *root, std::make_shared<const RuntimeEngineAdmission>(admission));
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Result<DeepSeekRuntimeArtifactAdmissionBinding>
DeepSeekRuntimeArtifactAdmissionBinding::Issue(
    const RuntimeEngineAdmission& admission,
    const DeepSeekDsparkRuntimeEvidenceManifest& evidence) {
  auto permit = issue_deepseek_dspark_runtime_permit(
      admission, evidence, evidence.artifact_root());
  if (!permit.ok()) return permit.status();
  auto root = compile_binding_root(
      admission, evidence.artifact_root(),
      evidence.domain_decision_roots_digest());
  if (!root.ok()) return root.status();
  return DeepSeekRuntimeArtifactAdmissionBinding(
      evidence.artifact_root(), evidence.domain_decision_roots_digest(),
      *root, std::make_shared<const RuntimeEngineAdmission>(admission));
}
#endif

Status DeepSeekRuntimeArtifactAdmissionBinding::validate(
    const RuntimeEngineAdmission& admission,
    const Sha256Digest& artifact_root) const {
  if (admission_anchor_ == nullptr || artifact_root != artifact_root_ ||
      evidence_manifest_root_ !=
          admission.evidence_roots().domain_decision_roots_digest) {
    return Status::FailedPrecondition(
        "DeepSeek runtime artifact admission binding was spliced");
  }
  auto root = compile_binding_root(
      admission, artifact_root, evidence_manifest_root_);
  if (!root.ok()) return root.status();
  if (*root != binding_root_) {
    return Status::FailedPrecondition(
        "DeepSeek runtime artifact admission binding differs");
  }
  return Status::Ok();
}

}  // namespace pih
