#include "pih/model/deepseek_rank_artifact_metadata_receipt.h"

#include <utility>

#include "pih/core/canonical_hash.h"
#include "pih/core/sha256.h"

namespace pih {

Result<Sha256Digest> compile_deepseek_rank_artifact_metadata_receipt_root(
    const DeepSeekRankArtifactTransferManifest& transfer_manifest,
    const Sha256Digest& adoption_receipt_root,
    const Sha256Digest& descriptor_transaction_root,
    const Sha256Digest& metadata_transaction_root,
    const Sha256Digest& metadata_root,
    const Sha256Digest& blob_sha256, std::uint64_t blob_bytes,
    bool dspark_enabled) {
  const auto& fields = transfer_manifest.fields();
  if (fields.world_size < 1 || fields.world_size > 4 ||
      fields.rank >= fields.world_size || blob_bytes == 0 ||
      transfer_manifest.manifest_root() == Sha256Digest{} ||
      adoption_receipt_root == Sha256Digest{} ||
      descriptor_transaction_root == Sha256Digest{} ||
      metadata_transaction_root == Sha256Digest{} ||
      metadata_root == Sha256Digest{} || blob_sha256 == Sha256Digest{}) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata receipt root input is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-metadata-receipt:v1", 12);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, fields.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, fields.worker_generation);
  if (status.ok()) status = builder->add_u32(3, fields.world_size);
  if (status.ok()) status = builder->add_u32(4, fields.rank);
  if (status.ok()) status = builder->add_hash(5, adoption_receipt_root);
  if (status.ok()) {
    status = builder->add_hash(6, transfer_manifest.manifest_root());
  }
  if (status.ok()) {
    status = builder->add_hash(7, descriptor_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(8, metadata_transaction_root);
  if (status.ok()) status = builder->add_hash(9, metadata_root);
  if (status.ok()) status = builder->add_hash(10, blob_sha256);
  if (status.ok()) status = builder->add_u64(11, blob_bytes);
  if (status.ok()) {
    status = builder->add_u32(12, dspark_enabled ? 1U : 0U);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

DeepSeekRankArtifactMetadataReceipt::DeepSeekRankArtifactMetadataReceipt(
    DeepSeekRankArtifactAdoptionReceipt adoption_receipt,
    DeepSeekRankArtifactMetadataBlob blob,
    Sha256Digest metadata_transaction_root,
    Sha256Digest blob_sha256, std::uint64_t blob_bytes,
    Sha256Digest receipt_root) noexcept
    : adoption_receipt_(std::move(adoption_receipt)),
      blob_(std::move(blob)),
      metadata_transaction_root_(metadata_transaction_root),
      blob_sha256_(blob_sha256), blob_bytes_(blob_bytes),
      receipt_root_(receipt_root) {}

Result<DeepSeekRankArtifactMetadataReceipt>
DeepSeekRankArtifactMetadataReceipt::Create(
    DeepSeekRankArtifactMetadataReceiver receiver) {
  if (!receiver.complete_ || receiver.poisoned_ || receiver.pending_ack_ ||
      !receiver.decoded_blob_ || !receiver.metadata_transaction_root_ ||
      !receiver.metadata_root_ || !receiver.blob_sha256_ ||
      !receiver.total_blob_bytes_ || !receiver.chunk_count_ ||
      receiver.next_chunk_index_ != *receiver.chunk_count_ ||
      receiver.reassembled_.size() != *receiver.total_blob_bytes_ ||
      !receiver.adoption_receipt_.retains_descriptor_owners()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata receiver is not sealed");
  }
  auto digest = sha256(receiver.reassembled_);
  if (!digest.ok()) return digest.status();
  if (*digest != *receiver.blob_sha256_ ||
      receiver.decoded_blob_->metadata_root() != *receiver.metadata_root_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata receipt content differs");
  }
  const auto& transfer = receiver.adoption_receipt_.transfer_manifest();
  auto receipt_root =
      compile_deepseek_rank_artifact_metadata_receipt_root(
          transfer, receiver.adoption_receipt_.receipt_root(),
          receiver.adoption_receipt_.transaction_root(),
          *receiver.metadata_transaction_root_, *receiver.metadata_root_,
          *receiver.blob_sha256_, *receiver.total_blob_bytes_,
          receiver.expected_dspark_enabled_);
  if (!receipt_root.ok()) return receipt_root.status();
  return DeepSeekRankArtifactMetadataReceipt(
      std::move(receiver.adoption_receipt_),
      std::move(*receiver.decoded_blob_),
      *receiver.metadata_transaction_root_, *receiver.blob_sha256_,
      *receiver.total_blob_bytes_, *receipt_root);
}

}  // namespace pih
