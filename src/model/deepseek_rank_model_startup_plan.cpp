#include "pih/model/deepseek_rank_model_startup_plan.h"

#include <stdexcept>
#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

Status validate_profile_join(
    const RuntimeEngineAdmission& admission,
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
    const RuntimeProfileReadinessReceipt& readiness,
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekPipelineCapacity& capacity) {
  if (!admission.production_ready() ||
      !admission.evidence_projection_bound() ||
      admission.model() != RuntimeProfileModel::kDeepSeekV4Flash0731 ||
      admission.weight_format() != RuntimeProfileWeightFormat::kDeepSeekNative ||
      admission.device_ordinals().size() != manifests.size() ||
      admission.dspark_enabled() != capacity.dspark_enabled ||
      bootstrap.deployment_generation() == 0 ||
      bootstrap.expected_envelope_root() != admission.envelope_root() ||
      readiness.deployment_generation() == 0 ||
      readiness.deployment_generation() != bootstrap.deployment_generation() ||
      readiness.authority_snapshot_root() !=
          bootstrap.authority_snapshot_root() ||
      readiness.bootstrap_manifest_root() != bootstrap.manifest_root() ||
      readiness.readiness_nonce_digest() !=
          bootstrap.readiness_nonce_digest() ||
      readiness.device_observation_root() !=
          admission.device_observation_root() ||
      readiness.capacity_plan_instance_root() !=
          supervisor.capacity_plan_instance_root() ||
      readiness.policy_digest() != admission.policy_digest() ||
      readiness.catalog_root() != admission.catalog_root() ||
      readiness.envelope_root() != admission.envelope_root() ||
      readiness.reference_closure_root() !=
          admission.reference_closure_root() ||
      readiness.graph_snapshot_root() != admission.graph_snapshot_root() ||
      !nonzero(readiness.authority_snapshot_root()) ||
      !nonzero(readiness.readiness_nonce_digest()) ||
      !nonzero(readiness.receipt_root())) {
    return Status::FailedPrecondition(
        "DeepSeek rank model profile readiness join is invalid");
  }
  for (std::size_t rank = 0; rank < manifests.size(); ++rank) {
    if (admission.device_ordinals()[rank] !=
        manifests[rank].startup_device_ordinal) {
      return Status::FailedPrecondition(
          "DeepSeek model startup device order differs from admission");
    }
  }
  return Status::Ok();
}

}  // namespace

DeepSeekRankModelStartupPlan::DeepSeekRankModelStartupPlan(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t world_size, std::uint64_t model_startup_deadline_ns,
    Sha256Digest manifest_root, Sha256Digest capacity_plan_instance_root,
    Sha256Digest profile_readiness_root,
    Sha256Digest post_exec_resource_seal_root,
    Sha256Digest pipeline_plan_root, Sha256Digest pipeline_capacity_root,
    Sha256Digest plan_root,
    std::vector<DeepSeekRankModelStartupSeed> rank_seeds,
    std::shared_ptr<const RuntimeEngineAdmission> admission_anchor) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      world_size_(world_size),
      model_startup_deadline_ns_(model_startup_deadline_ns),
      manifest_root_(manifest_root),
      capacity_plan_instance_root_(capacity_plan_instance_root),
      profile_readiness_root_(profile_readiness_root),
      post_exec_resource_seal_root_(post_exec_resource_seal_root),
      pipeline_plan_root_(pipeline_plan_root),
      pipeline_capacity_root_(pipeline_capacity_root), plan_root_(plan_root),
      rank_seeds_(std::move(rank_seeds)),
      admission_anchor_(std::move(admission_anchor)) {}

Result<DeepSeekRankModelStartupPlan> DeepSeekRankModelStartupPlan::Compile(
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
    const RuntimeProfileReadinessReceipt& profile_readiness,
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankPostExecResourceSeal& resource_seal,
    const DeepSeekPipelinePlan& pipeline,
    const DeepSeekPipelineCapacity& capacity,
    std::uint64_t model_startup_deadline_ns) {
  const auto* admission = supervisor.capacity_admission();
  if (admission == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank capacity admission authority is unavailable");
  }
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto pipeline_root = compile_deepseek_pipeline_plan_root(pipeline);
  if (!pipeline_root.ok()) return pipeline_root.status();
  auto capacity_root = compile_deepseek_pipeline_capacity_root(capacity);
  if (!capacity_root.ok()) return capacity_root.status();

  if (!supervisor.ready() || supervisor.failed() ||
      !supervisor.capacity_authority_retained() ||
      supervisor.manifest_root() != *manifest_root ||
      manifests.empty() || pipeline.world_size() != manifests.size() ||
      capacity.world_size != manifests.size() ||
      pipeline.rank(pipeline.world_size() - 1).owns_dspark !=
          capacity.dspark_enabled ||
      resource_seal.engine_epoch() != manifests[0].engine_epoch ||
      resource_seal.worker_generation() != manifests[0].worker_generation ||
      resource_seal.world_size() != manifests.size() ||
      resource_seal.capacity_plan_instance_root() !=
          supervisor.capacity_plan_instance_root() ||
      !nonzero(resource_seal.seal_root()) ||
      model_startup_deadline_ns <= manifests[0].startup_deadline_ns) {
    return Status::FailedPrecondition(
        "DeepSeek rank model startup generation join is invalid");
  }
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    if (manifests[rank].startup_deadline_ns !=
            manifests[0].startup_deadline_ns ||
        supervisor.exec_ready(rank) == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek rank model startup exec generation is incomplete");
    }
  }
  auto status = validate_profile_join(*admission, bootstrap, profile_readiness,
                                      supervisor, manifests, capacity);
  if (!status.ok()) return status;

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-model-startup-plan:v1", 15);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, manifests[0].engine_epoch);
  if (status.ok()) status = builder->add_u64(2, manifests[0].worker_generation);
  if (status.ok()) {
    status = builder->add_u32(
        3, static_cast<std::uint32_t>(manifests.size()));
  }
  if (status.ok()) status = builder->add_hash(4, *manifest_root);
  if (status.ok()) {
    status = builder->add_hash(5, supervisor.capacity_plan_instance_root());
  }
  if (status.ok()) status = builder->add_hash(6, bootstrap.manifest_root());
  if (status.ok()) status = builder->add_hash(7, profile_readiness.receipt_root());
  if (status.ok()) status = builder->add_hash(8, resource_seal.seal_root());
  if (status.ok()) status = builder->add_hash(9, *pipeline_root);
  if (status.ok()) status = builder->add_hash(10, *capacity_root);
  if (status.ok()) status = builder->add_hash(11, admission->envelope_root());
  if (status.ok()) {
    status = builder->add_hash(12, admission->graph_snapshot_root());
  }
  if (status.ok()) {
    status = builder->add_hash(13, admission->device_observation_root());
  }
  if (status.ok()) {
    status = builder->add_u32(
        14, static_cast<std::uint32_t>(admission->residency()));
  }
  if (status.ok()) status = builder->add_u64(15, model_startup_deadline_ns);
  if (!status.ok()) return status;
  auto plan_root = builder->finalize();
  if (!plan_root.ok()) return plan_root.status();

  std::vector<DeepSeekRankModelStartupSeed> rank_seeds;
  rank_seeds.reserve(manifests.size());
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    const auto* exec_ready = supervisor.exec_ready(rank);
    auto rank_builder = CanonicalHashBuilder::Create(
        "pih:deepseek-rank-model-startup-seed:v1", 10);
    if (!rank_builder.ok()) return rank_builder.status();
    status = rank_builder->add_hash(1, *plan_root);
    if (status.ok()) status = rank_builder->add_u32(2, rank);
    if (status.ok()) {
      status = rank_builder->add_u64(
          3, manifests[rank].process_manifest_identity);
    }
    if (status.ok()) {
      status = rank_builder->add_u64(4, exec_ready->receipt.process_identity);
    }
    if (status.ok()) {
      status = rank_builder->add_u64(5, exec_ready->receipt.pidfd_identity);
    }
    if (status.ok()) {
      status = rank_builder->add_u64(6, exec_ready->receipt.control_identity);
    }
    if (status.ok()) {
      status = rank_builder->add_u64(7, exec_ready->challenge_identity);
    }
    if (status.ok()) {
      status = rank_builder->add_u64(
          8, manifests[rank].physical_device_identity);
    }
    if (status.ok()) {
      status = rank_builder->add_hash(
          9, manifests[rank].physical_device_uuid_commitment);
    }
    if (status.ok()) {
      status = rank_builder->add_u32(
          10, static_cast<std::uint32_t>(
                  manifests[rank].startup_device_ordinal));
    }
    if (!status.ok()) return status;
    auto rank_root = rank_builder->finalize();
    if (!rank_root.ok()) return rank_root.status();
    rank_seeds.push_back(
        {rank,
         manifests[rank].process_manifest_identity,
         exec_ready->receipt.process_identity,
         exec_ready->receipt.pidfd_identity,
         exec_ready->receipt.control_identity,
         exec_ready->challenge_identity,
         manifests[rank].physical_device_identity,
         manifests[rank].physical_device_uuid_commitment,
         manifests[rank].startup_device_ordinal,
         *rank_root});
  }

  return DeepSeekRankModelStartupPlan(
      manifests[0].engine_epoch, manifests[0].worker_generation,
      static_cast<std::uint32_t>(manifests.size()),
      model_startup_deadline_ns, *manifest_root,
      supervisor.capacity_plan_instance_root(),
      profile_readiness.receipt_root(), resource_seal.seal_root(),
      *pipeline_root, *capacity_root, *plan_root, std::move(rank_seeds),
      std::make_shared<const RuntimeEngineAdmission>(*admission));
}

const Sha256Digest& DeepSeekRankModelStartupPlan::rank_seed_root(
    std::uint32_t rank) const {
  if (rank >= rank_seeds_.size()) {
    throw std::out_of_range("DeepSeek model startup rank seed");
  }
  return rank_seeds_[rank].seed_root;
}

const DeepSeekRankModelStartupSeed&
DeepSeekRankModelStartupPlan::rank_seed(std::uint32_t rank) const {
  if (rank >= rank_seeds_.size()) {
    throw std::out_of_range("DeepSeek model startup rank seed");
  }
  return rank_seeds_[rank];
}

}  // namespace pih
