#include "pih/model/deepseek_rank_artifact_transfer_plan.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

namespace pih {
namespace {

bool stage_owns(const DeepSeekStagePlan& stage,
                const DeepSeekRankTensorRecord& record) noexcept {
  switch (record.role) {
    case DeepSeekTensorRole::kEmbedding:
      return stage.owns_embedding && record.logical_layer == UINT32_MAX;
    case DeepSeekTensorRole::kFinalHead:
      return stage.owns_lm_head && record.logical_layer == UINT32_MAX;
    case DeepSeekTensorRole::kMainLayer:
      return record.logical_layer >= stage.layers.first_layer &&
             record.logical_layer <= stage.layers.last_layer;
    case DeepSeekTensorRole::kDspark:
      return stage.owns_dspark && record.logical_layer <= 2U;
  }
  return false;
}

Status validate_rank_join(
    const DeepSeekStagePlan& stage,
    const DeepSeekRankArtifactHandoffManifest& manifest,
    std::span<const DeepSeekWorkerShardDescriptor> descriptors,
    ArtifactImmutabilityMode immutability_mode) {
  const auto& mapping = manifest.mapping();
  const auto records = manifest.tensor_records();
  const auto expectations = manifest.descriptor_expectations();
  if (manifest.rank() != stage.rank || mapping.rank != stage.rank ||
      records.empty() || mapping.owned_tensor_count != records.size() ||
      expectations.empty() || expectations.size() != mapping.shards.size() ||
      descriptors.size() != expectations.size()) {
    return Status::FailedPrecondition(
        "DeepSeek rank artifact transfer ownership shape differs");
  }

  std::map<std::string_view, std::uint64_t, std::less<>> shard_bytes;
  for (std::size_t index = 0; index < expectations.size(); ++index) {
    const auto& expected = expectations[index];
    const auto& shard = mapping.shards[index];
    if (expected.ordinal != index || expected.shard_name != shard.shard_name ||
        expected.identity.file_bytes != shard.file_bytes ||
        expected.immutability_mode != immutability_mode ||
        descriptors[index].shard_name != expected.shard_name ||
        descriptors[index].descriptor.identity() != expected.identity ||
        !shard_bytes.emplace(expected.shard_name, shard.file_bytes).second) {
      return Status::FailedPrecondition(
          "DeepSeek rank artifact descriptor join differs");
    }
  }

  std::uint64_t logical_bytes = 0;
  for (const auto& record : records) {
    const auto shard = shard_bytes.find(record.shard_name);
    if (!stage_owns(stage, record) || shard == shard_bytes.end() ||
        record.file_begin >= record.file_end ||
        record.file_end > shard->second) {
      return Status::FailedPrecondition(
          "DeepSeek rank artifact tensor ownership differs from pipeline");
    }
    auto total = checked_add_u64(logical_bytes, record.tensor_bytes);
    if (!total.ok()) return total.status();
    logical_bytes = *total;
    const auto covered = std::ranges::any_of(
        mapping.intervals, [&](const DeepSeekMappedInterval& interval) {
          return interval.shard_name == record.shard_name &&
                 interval.file_begin <= record.file_begin &&
                 interval.file_end >= record.file_end;
        });
    if (!covered) {
      return Status::FailedPrecondition(
          "DeepSeek rank artifact tensor is outside mapped intervals");
    }
  }
  if (logical_bytes != mapping.logical_tensor_bytes) {
    return Status::FailedPrecondition(
        "DeepSeek rank artifact logical byte ledger differs");
  }
  return Status::Ok();
}

}  // namespace

DeepSeekRankArtifactTransferPlan::DeepSeekRankArtifactTransferPlan(
    DeepSeekRankModelStartupPlan model_startup,
    DeepSeekRuntimeArtifactAdmissionBinding artifact_binding,
    DeepSeekRankArtifactHandoffPlan handoff,
    std::vector<DeepSeekRankArtifactTransferManifest> manifests,
    Sha256Digest plan_root) noexcept
    : model_startup_(std::move(model_startup)),
      artifact_binding_(std::move(artifact_binding)),
      handoff_(std::move(handoff)), manifests_(std::move(manifests)),
      plan_root_(plan_root) {}

Result<DeepSeekRankArtifactTransferPlan>
DeepSeekRankArtifactTransferPlan::Compile(
    DeepSeekRankModelStartupPlan model_startup,
    DeepSeekRuntimeArtifactAdmissionBinding artifact_binding,
    DeepSeekRankArtifactHandoffPlan handoff) {
  const auto* admission = model_startup.admission_anchor_.get();
  if (admission == nullptr || !model_startup.admission_authority_retained() ||
      !artifact_binding.admission_authority_retained() ||
      model_startup.world_size() < 1 || model_startup.world_size() > 4 ||
      handoff.world_size() != model_startup.world_size() ||
      handoff.artifact_root() != artifact_binding.artifact_root()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer antecedent ownership differs");
  }
  auto binding_status = artifact_binding.validate(
      *admission, handoff.artifact_root());
  if (!binding_status.ok()) return binding_status;

  auto pipeline = DeepSeekPipelinePlan::Create(
      model_startup.world_size(), admission->dspark_enabled());
  if (!pipeline.ok()) return pipeline.status();
  auto pipeline_root = compile_deepseek_pipeline_plan_root(*pipeline);
  if (!pipeline_root.ok()) return pipeline_root.status();
  if (*pipeline_root != model_startup.pipeline_plan_root()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer pipeline differs from model startup");
  }

  std::vector<DeepSeekRankArtifactTransferManifest> manifests;
  manifests.reserve(model_startup.world_size());
  for (std::uint32_t rank = 0; rank < model_startup.world_size(); ++rank) {
    const auto& handoff_manifest = handoff.rank_manifest(rank);
    const auto descriptors = handoff.descriptors(rank);
    auto status = validate_rank_join(
        pipeline->rank(rank), handoff_manifest, descriptors,
        handoff.immutability_mode());
    if (!status.ok()) return status;
    const auto descriptor_count =
        static_cast<std::uint32_t>(descriptors.size());
    const auto tensor_count =
        static_cast<std::uint32_t>(handoff_manifest.tensor_records().size());
    const auto& seed = model_startup.rank_seed(rank);
    DeepSeekRankArtifactTransferManifestFields fields{
        model_startup.engine_epoch(),
        model_startup.worker_generation(),
        model_startup.world_size(),
        rank,
        seed.process_manifest_identity,
        seed.process_identity,
        seed.pidfd_identity,
        seed.control_identity,
        seed.challenge_identity,
        model_startup.model_startup_deadline_ns(),
        handoff.immutability_mode(),
        handoff.source_catalog_production_eligible(),
        descriptor_count,
        tensor_count,
        kDeepSeekRankArtifactTransferDescriptorBatchMaximum,
        (descriptor_count +
         kDeepSeekRankArtifactTransferDescriptorBatchMaximum - 1U) /
            kDeepSeekRankArtifactTransferDescriptorBatchMaximum,
        model_startup.plan_root(),
        seed.seed_root,
        model_startup.capacity_plan_instance_root(),
        artifact_binding.binding_root(),
        handoff.plan_root(),
        handoff_manifest.manifest_root(),
        handoff.artifact_root(),
        handoff.mapping_root()};
    auto manifest = DeepSeekRankArtifactTransferManifest::Create(fields);
    if (!manifest.ok()) return manifest.status();
    manifests.push_back(std::move(*manifest));
  }

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-transfer-plan:v1",
      14U + model_startup.world_size());
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, model_startup.engine_epoch());
  if (status.ok()) {
    status = builder->add_u64(2, model_startup.worker_generation());
  }
  if (status.ok()) status = builder->add_u32(3, model_startup.world_size());
  if (status.ok()) {
    status = builder->add_u64(4, model_startup.model_startup_deadline_ns());
  }
  if (status.ok()) status = builder->add_hash(5, model_startup.plan_root());
  if (status.ok()) {
    status = builder->add_hash(6, model_startup.capacity_plan_instance_root());
  }
  if (status.ok()) {
    status = builder->add_hash(7, model_startup.pipeline_plan_root());
  }
  if (status.ok()) {
    status = builder->add_hash(8, model_startup.pipeline_capacity_root());
  }
  if (status.ok()) status = builder->add_hash(9, artifact_binding.binding_root());
  if (status.ok()) status = builder->add_hash(10, handoff.plan_root());
  if (status.ok()) status = builder->add_hash(11, handoff.artifact_root());
  if (status.ok()) status = builder->add_hash(12, handoff.mapping_root());
  if (status.ok()) {
    status = builder->add_u32(
        13, static_cast<std::uint32_t>(handoff.immutability_mode()));
  }
  if (status.ok()) {
    status = builder->add_u32(
        14, handoff.source_catalog_production_eligible() ? 1U : 0U);
  }
  for (std::uint32_t rank = 0;
       status.ok() && rank < model_startup.world_size(); ++rank) {
    status = builder->add_hash(
        static_cast<std::uint16_t>(rank + 100U),
        manifests[rank].manifest_root());
  }
  if (!status.ok()) return status;
  auto plan_root = builder->finalize();
  if (!plan_root.ok()) return plan_root.status();
  return DeepSeekRankArtifactTransferPlan(
      std::move(model_startup), std::move(artifact_binding),
      std::move(handoff), std::move(manifests), *plan_root);
}

const DeepSeekRankArtifactTransferManifest&
DeepSeekRankArtifactTransferPlan::rank_manifest(std::uint32_t rank) const {
  if (rank >= manifests_.size()) {
    throw std::out_of_range("DeepSeek artifact transfer rank manifest");
  }
  return manifests_[rank];
}

std::array<std::byte, kDeepSeekRankArtifactTransferManifestBytes>
DeepSeekRankArtifactTransferPlan::encode_rank_manifest(
    std::uint32_t rank) const {
  return encode_deepseek_rank_artifact_transfer_manifest(rank_manifest(rank));
}

}  // namespace pih
