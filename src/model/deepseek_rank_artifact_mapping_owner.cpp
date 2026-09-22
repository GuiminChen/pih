#include "pih/model/deepseek_rank_artifact_mapping_owner.h"

#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {

Result<Sha256Digest> compile_deepseek_rank_artifact_mapping_owner_root(
    const DeepSeekRankArtifactTransferManifest& transfer_manifest,
    const DeepSeekRankArtifactMetadataBlob& blob,
    const Sha256Digest& metadata_receipt_root,
    const Sha256Digest& descriptor_transaction_root,
    const Sha256Digest& metadata_transaction_root) {
  const auto& fields = blob.fields();
  const auto& transfer_fields = transfer_manifest.fields();
  if (fields.world_size < 1 || fields.world_size > 4 ||
      fields.rank >= fields.world_size ||
      fields.engine_epoch != transfer_fields.engine_epoch ||
      fields.worker_generation != transfer_fields.worker_generation ||
      fields.world_size != transfer_fields.world_size ||
      fields.rank != transfer_fields.rank ||
      fields.transfer_manifest_root != transfer_manifest.manifest_root() ||
      metadata_receipt_root == Sha256Digest{} ||
      descriptor_transaction_root == Sha256Digest{} ||
      metadata_transaction_root == Sha256Digest{} ||
      blob.metadata_root() == Sha256Digest{} ||
      blob.tensor_records().empty() || blob.mapping().intervals.empty()) {
    return Status::InvalidArgument(
        "DeepSeek artifact mapping owner root input is invalid");
  }
  const auto tensor_count = static_cast<std::uint32_t>(
      blob.tensor_records().size());
  const auto interval_count = static_cast<std::uint32_t>(
      blob.mapping().intervals.size());
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-mapping-owner:v1", 19);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, fields.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, fields.worker_generation);
  if (status.ok()) status = builder->add_u32(3, fields.world_size);
  if (status.ok()) status = builder->add_u32(4, fields.rank);
  if (status.ok()) status = builder->add_hash(5, metadata_receipt_root);
  if (status.ok()) {
    status = builder->add_hash(6, descriptor_transaction_root);
  }
  if (status.ok()) {
    status = builder->add_hash(7, metadata_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(8, blob.metadata_root());
  if (status.ok()) status = builder->add_hash(9, fields.artifact_root);
  if (status.ok()) status = builder->add_hash(10, fields.mapping_root);
  if (status.ok()) status = builder->add_hash(11, fields.rank_mapping_root);
  if (status.ok()) status = builder->add_hash(12, fields.tensor_handoff_root);
  if (status.ok()) status = builder->add_u32(13, tensor_count);
  if (status.ok()) status = builder->add_u32(14, interval_count);
  if (status.ok()) {
    status = builder->add_u64(15, blob.mapping().logical_tensor_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(16, blob.mapping().mapped_interval_bytes);
  }
  if (status.ok()) {
    status = builder->add_u32(
        17, static_cast<std::uint32_t>(transfer_fields.immutability_mode));
  }
  if (status.ok()) {
    status = builder->add_u32(
        18, transfer_fields.source_catalog_production_eligible ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u32(19, fields.dspark_enabled ? 1U : 0U);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

DeepSeekRankArtifactMappingOwner::DeepSeekRankArtifactMappingOwner(
    DeepSeekStageMappedInventory inventory, std::uint64_t engine_epoch,
    std::uint64_t worker_generation, std::uint32_t world_size,
    std::uint32_t rank,
    ArtifactImmutabilityMode immutability_mode,
    bool source_catalog_production_eligible, bool dspark_enabled,
    Sha256Digest metadata_receipt_root,
    Sha256Digest descriptor_transaction_root,
    Sha256Digest metadata_transaction_root, Sha256Digest metadata_root,
    Sha256Digest mapping_owner_root,
    DeepSeekRankMappingPlan mapping_plan,
    std::vector<DeepSeekRankTensorRecord> tensor_records) noexcept
    : inventory_(std::move(inventory)),
      mapping_plan_(std::move(mapping_plan)),
      tensor_records_(std::move(tensor_records)),
      engine_epoch_(engine_epoch),
      worker_generation_(worker_generation), world_size_(world_size),
      rank_(rank),
      immutability_mode_(immutability_mode),
      source_catalog_production_eligible_(
          source_catalog_production_eligible),
      dspark_enabled_(dspark_enabled),
      metadata_receipt_root_(metadata_receipt_root),
      descriptor_transaction_root_(descriptor_transaction_root),
      metadata_transaction_root_(metadata_transaction_root),
      metadata_root_(metadata_root),
      mapping_owner_root_(mapping_owner_root) {}

Result<std::unique_ptr<DeepSeekRankArtifactMappingOwner>>
DeepSeekRankArtifactMappingOwner::Create(
    DeepSeekRankArtifactMetadataReceipt receipt) {
  if (!receipt.retains_descriptor_and_metadata_owners() ||
      receipt.receipt_root() == Sha256Digest{} ||
      receipt.adoption_receipt_.descriptors_.empty()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact mapping receipt lacks private owners");
  }
  const auto& blob = receipt.blob_;
  const auto& fields = blob.fields();
  const auto& transfer_fields =
      receipt.adoption_receipt_.transfer_manifest().fields();
  const auto metadata_receipt_root = receipt.receipt_root();
  auto owner_root = compile_deepseek_rank_artifact_mapping_owner_root(
      receipt.adoption_receipt_.transfer_manifest(), blob,
      metadata_receipt_root, receipt.descriptor_transaction_root(),
      receipt.metadata_transaction_root());
  if (!owner_root.ok()) return owner_root.status();

  DeepSeekRankMappingPlan mapping_plan = blob.mapping();
  std::vector<DeepSeekRankTensorRecord> tensor_records(
      blob.tensor_records().begin(), blob.tensor_records().end());
  auto inventory = DeepSeekStageMappedInventory::Create(
      mapping_plan,
      std::move(receipt.adoption_receipt_.descriptors_));
  if (!inventory.ok()) return inventory.status();
  auto owner = std::unique_ptr<DeepSeekRankArtifactMappingOwner>(
      new DeepSeekRankArtifactMappingOwner(
          std::move(*inventory), fields.engine_epoch,
          fields.worker_generation, fields.world_size, fields.rank,
          transfer_fields.immutability_mode,
          transfer_fields.source_catalog_production_eligible,
          fields.dspark_enabled, metadata_receipt_root,
          receipt.descriptor_transaction_root(),
          receipt.metadata_transaction_root(), receipt.metadata_root(),
          *owner_root, std::move(mapping_plan),
          std::move(tensor_records)));
  auto source = DeepSeekMappedTensorSource::Create(
      owner->inventory_, owner->tensor_records_);
  if (!source.ok()) return source.status();
  owner->tensor_source_.emplace(std::move(*source));
  return owner;
}

Result<DeepSeekMappedTensorSpan>
DeepSeekRankArtifactMappingOwner::resolve(
    std::string_view tensor_name) const {
  if (!tensor_source_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact mapping tensor source is absent");
  }
  return tensor_source_->resolve(tensor_name);
}

}  // namespace pih
