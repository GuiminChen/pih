#include "pih/model/deepseek_rank_capacity_plan_instance.h"

#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

constexpr std::uint64_t kMaximumCapacityTemplateBytes = 16ULL * 1024 * 1024;

bool nonzero(const Sha256Digest& value) noexcept {
  for (const auto byte : value.bytes) {
    if (byte != std::byte{0}) return true;
  }
  return false;
}

Status validate_admission(
    const RuntimeEngineAdmission& admission,
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
    std::span<const DeepSeekRankProcessManifest> manifests,
    std::uint64_t expected_controller_process_identity) {
  if (!admission.production_ready() ||
      !admission.evidence_projection_bound() ||
      admission.model() != RuntimeProfileModel::kDeepSeekV4Flash0731 ||
      admission.weight_format() != RuntimeProfileWeightFormat::kDeepSeekNative ||
      admission.device_ordinals().size() != manifests.size() ||
      admission.capacity_template_exact_bytes() == 0 ||
      admission.capacity_template_exact_bytes() >
          kMaximumCapacityTemplateBytes ||
      bootstrap.deployment_generation() == 0 ||
      bootstrap.expected_envelope_root() != admission.envelope_root() ||
      expected_controller_process_identity == 0 ||
      !nonzero(admission.graph_snapshot_root()) ||
      !nonzero(bootstrap.authority_snapshot_root())) {
    return Status::FailedPrecondition(
        "DeepSeek rank capacity authority is not production-ready");
  }
  for (std::size_t rank = 0; rank < manifests.size(); ++rank) {
    if (manifests[rank].startup_device_ordinal !=
        admission.device_ordinals()[rank]) {
      return Status::FailedPrecondition(
          "DeepSeek rank manifest differs from admitted device order");
    }
  }
  return Status::Ok();
}

}  // namespace

DeepSeekRankCapacityPlanInstance::DeepSeekRankCapacityPlanInstance(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t world_size,
    std::uint64_t expected_controller_process_identity,
    Sha256Digest manifest_root, Sha256Digest resource_plan_root,
    Sha256Digest instance_root,
    std::shared_ptr<const RuntimeEngineAdmission> admission_anchor) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      world_size_(world_size),
      expected_controller_process_identity_(
      expected_controller_process_identity),
      manifest_root_(manifest_root), resource_plan_root_(resource_plan_root),
      instance_root_(instance_root),
      admission_anchor_(std::move(admission_anchor)) {}

DeepSeekRankCapacityPlanInstance::DeepSeekRankCapacityPlanInstance(
    DeepSeekRankCapacityPlanInstance&& other) noexcept
    : engine_epoch_(std::exchange(other.engine_epoch_, 0)),
      worker_generation_(std::exchange(other.worker_generation_, 0)),
      world_size_(std::exchange(other.world_size_, 0)),
      expected_controller_process_identity_(
          std::exchange(other.expected_controller_process_identity_, 0)),
      manifest_root_(std::exchange(other.manifest_root_, Sha256Digest{})),
      resource_plan_root_(
          std::exchange(other.resource_plan_root_, Sha256Digest{})),
      instance_root_(std::exchange(other.instance_root_, Sha256Digest{})),
      admission_anchor_(std::move(other.admission_anchor_)) {}

DeepSeekRankCapacityPlanInstance&
DeepSeekRankCapacityPlanInstance::operator=(
    DeepSeekRankCapacityPlanInstance&& other) noexcept {
  if (this != &other) {
    engine_epoch_ = std::exchange(other.engine_epoch_, 0);
    worker_generation_ = std::exchange(other.worker_generation_, 0);
    world_size_ = std::exchange(other.world_size_, 0);
    expected_controller_process_identity_ =
        std::exchange(other.expected_controller_process_identity_, 0);
    manifest_root_ = std::exchange(other.manifest_root_, Sha256Digest{});
    resource_plan_root_ =
        std::exchange(other.resource_plan_root_, Sha256Digest{});
    instance_root_ = std::exchange(other.instance_root_, Sha256Digest{});
    admission_anchor_ = std::move(other.admission_anchor_);
  }
  return *this;
}

Result<DeepSeekRankCapacityPlanInstance>
DeepSeekRankCapacityPlanInstance::Compile(
    const RuntimeEngineAdmission& admission,
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& resource_plan,
    std::uint64_t expected_controller_process_identity) {
  auto status = DeepSeekRankSpawnPreflightReceipt::ValidatePlan(
      manifests, resource_plan);
  if (!status.ok()) return status;
  status = validate_admission(admission, bootstrap, manifests,
                              expected_controller_process_identity);
  if (!status.ok()) return status;
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto plan_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, resource_plan);
  if (!plan_root.ok()) return plan_root.status();

  const auto& evidence = admission.evidence_roots();
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-capacity-plan-instance:v1", 31);
  if (!builder.ok()) return builder.status();
  status = builder->add_ascii_utf8(1, admission.profile_id());
  if (status.ok()) status = builder->add_ascii_utf8(2, admission.revision());
  if (status.ok())
    status = builder->add_u32(3, static_cast<std::uint32_t>(admission.model()));
  if (status.ok())
    status = builder->add_u32(
        4, static_cast<std::uint32_t>(admission.weight_format()));
  if (status.ok())
    status = builder->add_u32(
        5, static_cast<std::uint32_t>(admission.gpu_family()));
  if (status.ok()) status = builder->add_u32(6, admission.dspark_enabled());
  if (status.ok())
    status = builder->add_u32(
        7, static_cast<std::uint32_t>(admission.residency()));
  if (status.ok())
    status = builder->add_u32(
        8, static_cast<std::uint32_t>(manifests.size()));
  if (status.ok()) status = builder->add_hash(9, admission.policy_digest());
  if (status.ok()) status = builder->add_hash(10, admission.catalog_root());
  if (status.ok()) status = builder->add_hash(11, admission.envelope_root());
  if (status.ok())
    status = builder->add_hash(12, admission.reference_closure_root());
  if (status.ok())
    status = builder->add_hash(13, admission.graph_snapshot_root());
  if (status.ok())
    status = builder->add_hash(
        14, admission.reference_roots().capacity_template_root);
  if (status.ok())
    status = builder->add_u64(
        15, admission.capacity_template_exact_bytes());
  if (status.ok())
    status = builder->add_hash(16, admission.device_observation_root());
  if (status.ok()) status = builder->add_hash(17, evidence.bundle_root);
  if (status.ok()) status = builder->add_hash(18, evidence.audit_closure_root);
  if (status.ok())
    status = builder->add_hash(19, evidence.required_role_schema_root);
  if (status.ok())
    status = builder->add_hash(20, evidence.domain_decision_roots_digest);
  if (status.ok()) status = builder->add_hash(21, evidence.builder_policy_root);
  if (status.ok()) status = builder->add_hash(22, bootstrap.manifest_root());
  if (status.ok())
    status = builder->add_hash(23, bootstrap.inherited_fds().manifest_root());
  if (status.ok())
    status = builder->add_u64(24, bootstrap.deployment_generation());
  if (status.ok())
    status = builder->add_hash(25, bootstrap.authority_snapshot_root());
  if (status.ok())
    status = builder->add_hash(26, bootstrap.readiness_nonce_digest());
  if (status.ok())
    status = builder->add_u64(27, expected_controller_process_identity);
  if (status.ok()) status = builder->add_u64(28, manifests[0].engine_epoch);
  if (status.ok())
    status = builder->add_u64(29, manifests[0].worker_generation);
  if (status.ok()) status = builder->add_hash(30, *manifest_root);
  if (status.ok()) status = builder->add_hash(31, *plan_root);
  if (!status.ok()) return status;
  auto instance_root = builder->finalize();
  if (!instance_root.ok()) return instance_root.status();
  return DeepSeekRankCapacityPlanInstance(
      manifests[0].engine_epoch, manifests[0].worker_generation,
      static_cast<std::uint32_t>(manifests.size()),
      expected_controller_process_identity, *manifest_root, *plan_root,
      *instance_root,
      std::make_shared<const RuntimeEngineAdmission>(admission));
}

Status DeepSeekRankCapacityPlanInstance::validate_static_binding(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& resource_plan) const {
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto plan_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, resource_plan);
  if (!plan_root.ok()) return plan_root.status();
  if (manifests.empty() || engine_epoch_ != manifests[0].engine_epoch ||
      worker_generation_ != manifests[0].worker_generation ||
      world_size_ != manifests.size() || manifest_root_ != *manifest_root ||
      resource_plan_root_ != *plan_root ||
      expected_controller_process_identity_ == 0 ||
      !nonzero(instance_root_) || admission_anchor_ == nullptr ||
      !admission_anchor_->production_ready()) {
    return Status::FailedPrecondition(
        "DeepSeek rank capacity instance differs from static launch plan");
  }
  return Status::Ok();
}

}  // namespace pih
