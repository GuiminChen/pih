#include "pih/model/deepseek_rank_artifact_adoption_receipt.h"

#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {

Result<Sha256Digest> compile_deepseek_rank_artifact_adoption_receipt_root(
    const DeepSeekRankArtifactTransferManifest& transfer_manifest,
    const Sha256Digest& descriptor_transaction_root,
    const Sha256Digest& adopted_descriptor_set_root) {
  const auto& fields = transfer_manifest.fields();
  if (fields.world_size < 1 || fields.world_size > 4 ||
      fields.rank >= fields.world_size || fields.descriptor_count == 0 ||
      transfer_manifest.manifest_root() == Sha256Digest{} ||
      descriptor_transaction_root == Sha256Digest{} ||
      adopted_descriptor_set_root == Sha256Digest{}) {
    return Status::InvalidArgument(
        "DeepSeek artifact adoption receipt root input is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-adoption-receipt:v1", 19);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, fields.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, fields.worker_generation);
  if (status.ok()) status = builder->add_u32(3, fields.world_size);
  if (status.ok()) status = builder->add_u32(4, fields.rank);
  if (status.ok()) {
    status = builder->add_u64(5, fields.process_manifest_identity);
  }
  if (status.ok()) status = builder->add_u64(6, fields.process_identity);
  if (status.ok()) status = builder->add_u64(7, fields.pidfd_identity);
  if (status.ok()) status = builder->add_u64(8, fields.control_identity);
  if (status.ok()) status = builder->add_u64(9, fields.challenge_identity);
  if (status.ok()) {
    status = builder->add_hash(10, transfer_manifest.manifest_root());
  }
  if (status.ok()) {
    status = builder->add_hash(11, fields.artifact_admission_binding_root);
  }
  if (status.ok()) {
    status = builder->add_hash(12, descriptor_transaction_root);
  }
  if (status.ok()) {
    status = builder->add_hash(13, adopted_descriptor_set_root);
  }
  if (status.ok()) status = builder->add_u32(14, fields.descriptor_count);
  if (status.ok()) {
    status = builder->add_hash(15, fields.artifact_handoff_rank_root);
  }
  if (status.ok()) status = builder->add_hash(16, fields.artifact_root);
  if (status.ok()) status = builder->add_hash(17, fields.mapping_root);
  if (status.ok()) {
    status = builder->add_u32(
        18, static_cast<std::uint32_t>(fields.immutability_mode));
  }
  if (status.ok()) {
    status = builder->add_u32(
        19, fields.source_catalog_production_eligible ? 1U : 0U);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

DeepSeekRankArtifactAdoptionReceipt::DeepSeekRankArtifactAdoptionReceipt(
    DeepSeekRankArtifactTransferManifest transfer_manifest,
    std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations,
    std::vector<DeepSeekWorkerShardDescriptor> descriptors,
    std::int32_t control_fd,
    Sha256Digest transaction_root,
    Sha256Digest adopted_descriptor_set_root,
    Sha256Digest receipt_root) noexcept
    : transfer_manifest_(std::move(transfer_manifest)),
      expectations_(std::move(expectations)),
      descriptors_(std::move(descriptors)),
      control_fd_(control_fd),
      transaction_root_(transaction_root),
      adopted_descriptor_set_root_(adopted_descriptor_set_root),
      receipt_root_(receipt_root) {}

Result<DeepSeekRankArtifactAdoptionReceipt>
DeepSeekRankArtifactAdoptionReceipt::Create(
    DeepSeekRankArtifactTransferReceiver receiver) {
  if (!receiver.complete_ || receiver.poisoned_ ||
      !receiver.transfer_manifest_ || !receiver.transaction_root_ ||
      receiver.pending_ack_ || receiver.in_flight_lease_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact receiver is not sealed for adoption");
  }
  const auto& manifest = *receiver.transfer_manifest_;
  const auto& fields = manifest.fields();
  if (fields.descriptor_count == 0 ||
      receiver.adopted_expectations_.size() != fields.descriptor_count ||
      receiver.adopted_descriptors_.size() != fields.descriptor_count ||
      receiver.next_batch_index_ != fields.descriptor_batch_count) {
    return Status::FailedPrecondition(
        "DeepSeek artifact adoption inventory is incomplete");
  }
  for (std::size_t index = 0;
       index < receiver.adopted_expectations_.size(); ++index) {
    const auto& expectation = receiver.adopted_expectations_[index];
    const auto& descriptor = receiver.adopted_descriptors_[index];
    if (expectation.ordinal != index ||
        expectation.immutability_mode != fields.immutability_mode ||
        descriptor.shard_name != expectation.shard_name ||
        descriptor.descriptor.identity() != expectation.identity) {
      return Status::FailedPrecondition(
          "DeepSeek artifact adoption descriptor identity differs");
    }
  }
  auto adopted_root =
      compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
          fields.rank, receiver.adopted_expectations_);
  if (!adopted_root.ok()) return adopted_root.status();

  auto receipt_root =
      compile_deepseek_rank_artifact_adoption_receipt_root(
          manifest, *receiver.transaction_root_, *adopted_root);
  if (!receipt_root.ok()) return receipt_root.status();
  return DeepSeekRankArtifactAdoptionReceipt(
      std::move(*receiver.transfer_manifest_),
      std::move(receiver.adopted_expectations_),
      std::move(receiver.adopted_descriptors_), receiver.control_fd_,
      *receiver.transaction_root_,
      *adopted_root, *receipt_root);
}

}  // namespace pih
