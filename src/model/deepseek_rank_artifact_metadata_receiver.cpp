#include "pih/model/deepseek_rank_artifact_metadata_receiver.h"

#include <new>
#include <utility>

#include "pih/core/sha256.h"

namespace pih {

DeepSeekRankArtifactMetadataReceiver::DeepSeekRankArtifactMetadataReceiver(
    DeepSeekRankArtifactAdoptionReceipt adoption_receipt,
    bool expected_dspark_enabled,
    std::uint64_t maximum_reassembly_bytes,
    DeepSeekRankArtifactMetadataReceiverOperations& operations) noexcept
    : adoption_receipt_(std::move(adoption_receipt)),
      expected_dspark_enabled_(expected_dspark_enabled),
      maximum_reassembly_bytes_(maximum_reassembly_bytes),
      operations_(&operations) {}

Result<DeepSeekRankArtifactMetadataReceiver>
DeepSeekRankArtifactMetadataReceiver::Create(
    DeepSeekRankArtifactAdoptionReceipt adoption_receipt,
    bool expected_dspark_enabled,
    std::uint64_t maximum_reassembly_bytes,
    DeepSeekRankArtifactMetadataReceiverOperations& operations) {
  const auto& manifest = adoption_receipt.transfer_manifest();
  const auto& fields = manifest.fields();
  if (!adoption_receipt.retains_descriptor_owners() ||
      adoption_receipt.control_fd_ < 0 || fields.engine_epoch == 0 ||
      fields.worker_generation == 0 || fields.world_size < 1 ||
      fields.world_size > 4 || fields.rank >= fields.world_size ||
      fields.descriptor_count != adoption_receipt.descriptor_count() ||
      fields.deadline_ns == 0 ||
      adoption_receipt.transaction_root() == Sha256Digest{} ||
      maximum_reassembly_bytes == 0 ||
      maximum_reassembly_bytes >
          kDeepSeekRankArtifactMetadataBlobMaximumBytes) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata receiver identity is invalid");
  }
  return DeepSeekRankArtifactMetadataReceiver(
      std::move(adoption_receipt), expected_dspark_enabled,
      maximum_reassembly_bytes, operations);
}

Status DeepSeekRankArtifactMetadataReceiver::fail(Status cause) noexcept {
  poisoned_ = true;
  complete_ = false;
  if (cause.ok()) {
    return Status::Internal(
        "DeepSeek artifact metadata receiver failed");
  }
  return cause;
}

Status DeepSeekRankArtifactMetadataReceiver::ensure_before_deadline() {
  auto now = operations_->monotonic_now_ns();
  if (!now.ok()) {
    return Status::Internal(
        "DeepSeek artifact metadata receiver clock failed");
  }
  if (*now >= wait_deadline_ns()) {
    return Status::DeadlineExceeded(
        "DeepSeek artifact metadata receiver deadline expired");
  }
  return Status::Ok();
}

Status DeepSeekRankArtifactMetadataReceiver::validate_complete_blob(
    const DeepSeekRankArtifactMetadataChunkFields& final_fields) {
  if (reassembled_.size() != final_fields.total_blob_bytes ||
      reassembled_.size() != *total_blob_bytes_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata reassembled size differs");
  }
  auto digest = sha256(reassembled_);
  if (!digest.ok()) return digest.status();
  if (*digest != final_fields.blob_sha256 ||
      *digest != *blob_sha256_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata reassembled digest differs");
  }
  auto blob = decode_deepseek_rank_artifact_metadata_blob(reassembled_);
  if (!blob.ok()) return blob.status();
  const auto& blob_fields = blob->fields();
  const auto& transfer = adoption_receipt_.transfer_manifest();
  const auto& transfer_fields = transfer.fields();
  if (blob_fields.engine_epoch != transfer_fields.engine_epoch ||
      blob_fields.worker_generation != transfer_fields.worker_generation ||
      blob_fields.world_size != transfer_fields.world_size ||
      blob_fields.rank != transfer_fields.rank ||
      blob_fields.dspark_enabled != expected_dspark_enabled_ ||
      blob_fields.process_manifest_identity !=
          transfer_fields.process_manifest_identity ||
      blob_fields.process_identity != transfer_fields.process_identity ||
      blob_fields.pidfd_identity != transfer_fields.pidfd_identity ||
      blob_fields.control_identity != transfer_fields.control_identity ||
      blob_fields.challenge_identity != transfer_fields.challenge_identity ||
      blob_fields.descriptor_count != transfer_fields.descriptor_count ||
      blob_fields.transfer_manifest_root != transfer.manifest_root() ||
      blob_fields.artifact_admission_binding_root !=
          transfer_fields.artifact_admission_binding_root ||
      blob_fields.descriptor_transfer_transaction_root !=
          adoption_receipt_.transaction_root() ||
      blob_fields.artifact_handoff_rank_root !=
          transfer_fields.artifact_handoff_rank_root ||
      blob_fields.artifact_root != transfer_fields.artifact_root ||
      blob_fields.mapping_root != transfer_fields.mapping_root ||
      blob->metadata_root() != final_fields.metadata_root ||
      blob->metadata_root() != *metadata_root_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata blob differs from descriptor adoption");
  }
  decoded_blob_.emplace(std::move(*blob));
  return Status::Ok();
}

Status DeepSeekRankArtifactMetadataReceiver::accept_chunk(
    DeepSeekRankArtifactMetadataChunk chunk) {
  if (pending_ack_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata receiver already has a pending ACK");
  }
  const auto& fields = chunk.fields();
  const auto& transfer = adoption_receipt_.transfer_manifest();
  const auto& transfer_fields = transfer.fields();
  if (fields.engine_epoch != transfer_fields.engine_epoch ||
      fields.worker_generation != transfer_fields.worker_generation ||
      fields.world_size != transfer_fields.world_size ||
      fields.rank != transfer_fields.rank ||
      fields.chunk_index != next_chunk_index_ ||
      fields.process_manifest_identity !=
          transfer_fields.process_manifest_identity ||
      fields.process_identity != transfer_fields.process_identity ||
      fields.pidfd_identity != transfer_fields.pidfd_identity ||
      fields.control_identity != transfer_fields.control_identity ||
      fields.challenge_identity != transfer_fields.challenge_identity ||
      fields.transfer_manifest_root != transfer.manifest_root() ||
      fields.descriptor_transfer_transaction_root !=
          adoption_receipt_.transaction_root() ||
      fields.payload_offset != reassembled_.size()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata chunk differs from descriptor adoption");
  }

  if (!metadata_transaction_root_) {
    if (fields.chunk_index != 0 || fields.total_blob_bytes == 0 ||
        fields.total_blob_bytes > maximum_reassembly_bytes_) {
      return Status::ResourceExhausted(
          "DeepSeek artifact metadata reassembly exceeds its authority");
    }
    metadata_transaction_root_ = fields.metadata_transaction_root;
    metadata_root_ = fields.metadata_root;
    blob_sha256_ = fields.blob_sha256;
    total_blob_bytes_ = fields.total_blob_bytes;
    chunk_count_ = fields.chunk_count;
    try {
      reassembled_.reserve(
          static_cast<std::size_t>(fields.total_blob_bytes));
    } catch (const std::bad_alloc&) {
      return Status::ResourceExhausted(
          "DeepSeek artifact metadata reassembly allocation failed");
    } catch (const std::length_error&) {
      return Status::ResourceExhausted(
          "DeepSeek artifact metadata reassembly length is invalid");
    }
  } else if (fields.metadata_transaction_root !=
                 *metadata_transaction_root_ ||
             fields.metadata_root != *metadata_root_ ||
             fields.blob_sha256 != *blob_sha256_ ||
             fields.total_blob_bytes != *total_blob_bytes_ ||
             fields.chunk_count != *chunk_count_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata chunk stream identity changed");
  }

  try {
    reassembled_.insert(reassembled_.end(), chunk.payload().begin(),
                        chunk.payload().end());
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted(
        "DeepSeek artifact metadata reassembly append failed");
  } catch (const std::length_error&) {
    return Status::ResourceExhausted(
        "DeepSeek artifact metadata reassembly length overflowed");
  }
  const auto final_chunk = fields.chunk_index + 1U == fields.chunk_count;
  if (final_chunk) {
    auto status = validate_complete_blob(fields);
    if (!status.ok()) return status;
  }

  DeepSeekRankArtifactMetadataChunkAckFields ack_fields{
      fields.engine_epoch,
      fields.worker_generation,
      fields.world_size,
      fields.rank,
      fields.chunk_index,
      fields.chunk_count,
      fields.payload_offset + fields.payload_bytes,
      fields.total_blob_bytes,
      fields.process_manifest_identity,
      fields.process_identity,
      fields.pidfd_identity,
      fields.control_identity,
      fields.challenge_identity,
      fields.transfer_manifest_root,
      fields.descriptor_transfer_transaction_root,
      fields.metadata_root,
      fields.metadata_transaction_root,
      fields.blob_sha256,
      fields.chunk_sha256};
  auto ack = DeepSeekRankArtifactMetadataChunkAck::Create(ack_fields);
  if (!ack.ok()) return ack.status();
  pending_ack_ = encode_deepseek_rank_artifact_metadata_chunk_ack(*ack);
  pending_ack_is_final_ = final_chunk;
  return Status::Ok();
}

Status DeepSeekRankArtifactMetadataReceiver::flush_ack() {
  if (!pending_ack_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata receiver ACK is absent");
  }
  auto status = ensure_before_deadline();
  if (!status.ok()) return status;
  status = operations_->send_ack(
      adoption_receipt_.control_fd_, *pending_ack_);
  if (!status.ok()) return status;
  status = ensure_before_deadline();
  if (!status.ok()) return status;
  pending_ack_.reset();
  ++next_chunk_index_;
  if (pending_ack_is_final_) {
    if (!decoded_blob_ || !chunk_count_ ||
        next_chunk_index_ != *chunk_count_) {
      return Status::FailedPrecondition(
          "DeepSeek artifact metadata final ACK state is incomplete");
    }
    complete_ = true;
  }
  pending_ack_is_final_ = false;
  return Status::Ok();
}

Status DeepSeekRankArtifactMetadataReceiver::advance() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata receiver is poisoned");
  }
  if (complete_) return Status::Ok();
  if (pending_ack_) {
    auto status = flush_ack();
    if (!status.ok()) {
      if (status.code() == StatusCode::kUnavailable) return status;
      return fail(status);
    }
    if (complete_) return Status::Ok();
  }
  auto status = ensure_before_deadline();
  if (!status.ok()) return fail(status);
  auto frame = operations_->receive_chunk(adoption_receipt_.control_fd_);
  if (!frame.ok()) {
    if (frame.status().code() == StatusCode::kUnavailable) {
      return frame.status();
    }
    return fail(frame.status());
  }
  if (!frame->has_value()) {
    return Status::Unavailable(
        "DeepSeek artifact metadata chunk is pending");
  }
  status = ensure_before_deadline();
  if (!status.ok()) return fail(status);
  auto chunk = decode_deepseek_rank_artifact_metadata_chunk(**frame);
  if (!chunk.ok()) return fail(chunk.status());
  status = accept_chunk(std::move(*chunk));
  if (!status.ok()) return fail(status);
  status = flush_ack();
  if (!status.ok()) {
    if (status.code() == StatusCode::kUnavailable) return status;
    return fail(status);
  }
  return complete_ ? Status::Ok()
                   : Status::Unavailable(
                         "DeepSeek artifact metadata receiver is pending");
}

}  // namespace pih
