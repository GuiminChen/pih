#include "pih/model/deepseek_rank_artifact_metadata_transfer_transaction.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "pih/core/canonical_hash.h"
#include "pih/core/sha256.h"
#include "pih/model/deepseek_rank_artifact_adoption_receipt.h"
#include "pih/model/deepseek_rank_artifact_mapping_owner.h"
#include "pih/model/deepseek_rank_artifact_metadata_receipt.h"

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

Result<Sha256Digest> compile_rank_transaction_root(
    std::uint32_t rank,
    const DeepSeekRankArtifactTransferManifest& transfer_manifest,
    const Sha256Digest& metadata_root, const Sha256Digest& blob_sha256,
    std::uint64_t blob_bytes, std::uint32_t chunk_count) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-metadata-transfer-transaction-rank:v1",
      7);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, rank);
  if (status.ok()) {
    status = builder->add_hash(2, transfer_manifest.manifest_root());
  }
  if (status.ok()) {
    status = builder->add_hash(
        3, transfer_manifest.fields().artifact_handoff_rank_root);
  }
  if (status.ok()) status = builder->add_hash(4, metadata_root);
  if (status.ok()) status = builder->add_hash(5, blob_sha256);
  if (status.ok()) status = builder->add_u64(6, blob_bytes);
  if (status.ok()) status = builder->add_u32(7, chunk_count);
  if (!status.ok()) return status;
  return builder->finalize();
}

std::uint32_t chunk_count_for(std::size_t blob_bytes) {
  return static_cast<std::uint32_t>(
      (blob_bytes +
       kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes - 1U) /
      kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes);
}

}  // namespace

DeepSeekRankArtifactMetadataTransferTransaction::
    DeepSeekRankArtifactMetadataTransferTransaction(
        DeepSeekRankArtifactTransferTransaction descriptor_transaction,
        DeepSeekRankArtifactMetadataTransferOperations& operations,
        std::vector<RankPayload> payloads,
        DeepSeekNodeArtifactPrefaultLayout node_prefault_layout,
        Sha256Digest transaction_root) noexcept
    : descriptor_transaction_(std::move(descriptor_transaction)),
      operations_(&operations), payloads_(std::move(payloads)),
      ranks_(descriptor_transaction_.world_size()),
      node_prefault_layout_(std::move(node_prefault_layout)),
      transaction_root_(transaction_root) {}

Result<DeepSeekRankArtifactMetadataTransferTransaction>
DeepSeekRankArtifactMetadataTransferTransaction::Create(
    DeepSeekRankArtifactTransferTransaction descriptor_transaction,
    DeepSeekRankArtifactMetadataTransferOperations& operations) {
  if (!descriptor_transaction.complete() ||
      descriptor_transaction.poisoned() ||
      !descriptor_transaction.retains_transfer_antecedents() ||
      descriptor_transaction.world_size() < 1 ||
      descriptor_transaction.world_size() > 4 ||
      !nonzero(descriptor_transaction.transaction_root())) {
    return Status::FailedPrecondition(
        "DeepSeek metadata transfer requires a completed descriptor transaction");
  }
  const auto& plan = descriptor_transaction.plan_;
  const auto& generation = plan.rank_manifest(0).fields();
  const auto fail_create =
      [&](Status cause)
      -> Result<DeepSeekRankArtifactMetadataTransferTransaction> {
    if (cause.ok()) {
      cause = Status::Internal(
          "DeepSeek artifact metadata transfer construction failed");
    }
    (void)operations.abort_generation(
        generation.engine_epoch, generation.worker_generation, cause);
    return cause;
  };
  std::vector<RankPayload> payloads;
  std::vector<Sha256Digest> rank_roots;
  std::vector<Sha256Digest> adoption_receipt_roots;
  std::vector<DeepSeekRankArtifactMetadataBlob> semantic_blobs;
  std::vector<DeepSeekRankMappingPlan> rank_mapping_plans;
  payloads.reserve(plan.world_size());
  rank_roots.reserve(plan.world_size());
  adoption_receipt_roots.reserve(plan.world_size());
  semantic_blobs.reserve(plan.world_size());
  rank_mapping_plans.reserve(plan.world_size());
  const auto deadline = plan.rank_manifest(0).fields().deadline_ns;

  for (std::uint32_t rank = 0; rank < plan.world_size(); ++rank) {
    const auto& transfer_manifest = plan.rank_manifest(rank);
    const auto& transfer_fields = transfer_manifest.fields();
    const auto& handoff_manifest = plan.handoff_.rank_manifest(rank);
    auto prefault_layout = compile_deepseek_rank_artifact_prefault_layout(
        handoff_manifest.mapping());
    if (!prefault_layout.ok()) return fail_create(prefault_layout.status());
    const auto mapping_root = compile_deepseek_rank_mapping_plan_root(
        handoff_manifest.mapping());
    if (!mapping_root.ok()) return fail_create(mapping_root.status());
    const auto descriptor_root =
        compile_deepseek_rank_artifact_descriptor_handoff_root(
            handoff_manifest.descriptor_expectations());
    if (!descriptor_root.ok()) return fail_create(descriptor_root.status());
    const auto tensor_root = compile_deepseek_rank_tensor_handoff_root(
        handoff_manifest.tensor_records());
    if (!tensor_root.ok()) return fail_create(tensor_root.status());
    if (transfer_fields.rank != rank ||
        transfer_fields.world_size != plan.world_size() ||
        transfer_fields.deadline_ns != deadline ||
        transfer_fields.descriptor_count !=
            handoff_manifest.descriptor_expectations().size() ||
        transfer_fields.tensor_record_count !=
            handoff_manifest.tensor_records().size() ||
        transfer_fields.artifact_handoff_rank_root !=
            handoff_manifest.manifest_root()) {
      return fail_create(Status::FailedPrecondition(
          "DeepSeek metadata transfer rank inventory drifted"));
    }
    DeepSeekRankArtifactMetadataBlobFields blob_fields{
        transfer_fields.engine_epoch,
        transfer_fields.worker_generation,
        transfer_fields.world_size,
        rank,
        plan.dspark_enabled(),
        transfer_fields.process_manifest_identity,
        transfer_fields.process_identity,
        transfer_fields.pidfd_identity,
        transfer_fields.control_identity,
        transfer_fields.challenge_identity,
        transfer_fields.descriptor_count,
        transfer_manifest.manifest_root(),
        transfer_fields.artifact_admission_binding_root,
        descriptor_transaction.transaction_root(),
        transfer_fields.artifact_handoff_rank_root,
        transfer_fields.artifact_root,
        transfer_fields.mapping_root,
        *mapping_root,
        *descriptor_root,
        *tensor_root};
    auto blob = DeepSeekRankArtifactMetadataBlob::Create(
        std::move(blob_fields), handoff_manifest.mapping(),
        std::vector<DeepSeekRankTensorRecord>(
            handoff_manifest.tensor_records().begin(),
            handoff_manifest.tensor_records().end()));
    if (!blob.ok()) return fail_create(blob.status());
    const auto metadata_root = blob->metadata_root();
    auto encoded = encode_deepseek_rank_artifact_metadata_blob(*blob);
    if (!encoded.ok()) return fail_create(encoded.status());
    auto encoded_sha256 = sha256(*encoded);
    if (!encoded_sha256.ok()) return fail_create(encoded_sha256.status());
    auto adopted_descriptor_set_root =
        compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
            rank, handoff_manifest.descriptor_expectations());
    if (!adopted_descriptor_set_root.ok()) {
      return fail_create(adopted_descriptor_set_root.status());
    }
    auto adoption_receipt_root =
        compile_deepseek_rank_artifact_adoption_receipt_root(
            transfer_manifest, descriptor_transaction.transaction_root(),
            *adopted_descriptor_set_root);
    if (!adoption_receipt_root.ok()) {
      return fail_create(adoption_receipt_root.status());
    }
    const auto chunk_count = chunk_count_for(encoded->size());
    if (chunk_count == 0 ||
        encoded->size() > kDeepSeekRankArtifactMetadataBlobMaximumBytes) {
      return fail_create(Status::ResourceExhausted(
          "DeepSeek metadata transfer blob exceeds its bound"));
    }
    auto rank_root = compile_rank_transaction_root(
        rank, transfer_manifest, metadata_root, *encoded_sha256,
        encoded->size(), chunk_count);
    if (!rank_root.ok()) return fail_create(rank_root.status());
    rank_roots.push_back(*rank_root);
    adoption_receipt_roots.push_back(*adoption_receipt_root);
    semantic_blobs.push_back(std::move(*blob));
    payloads.push_back(
        {std::move(*encoded), metadata_root, *encoded_sha256, {},
         *prefault_layout,
         handoff_manifest.mapping().mapped_interval_bytes,
         transfer_fields.immutability_mode,
         transfer_fields.source_catalog_production_eligible,
         chunk_count});
    rank_mapping_plans.push_back(handoff_manifest.mapping());
  }

  auto node_prefault_layout =
      compile_deepseek_node_artifact_prefault_layout(rank_mapping_plans);
  if (!node_prefault_layout.ok()) {
    return fail_create(node_prefault_layout.status());
  }

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-metadata-transfer-transaction:v1",
      plan.world_size() + 6U);
  if (!builder.ok()) return fail_create(builder.status());
  auto status = builder->add_hash(
      1, descriptor_transaction.transaction_root());
  if (status.ok()) status = builder->add_hash(2, plan.plan_root());
  if (status.ok()) status = builder->add_hash(3, plan.artifact_root());
  if (status.ok()) status = builder->add_u32(4, plan.world_size());
  if (status.ok()) status = builder->add_u64(5, deadline);
  if (status.ok()) {
    status = builder->add_hash(6, node_prefault_layout->layout_root);
  }
  for (std::uint32_t rank = 0;
       status.ok() && rank < plan.world_size(); ++rank) {
    status = builder->add_hash(
        static_cast<std::uint16_t>(rank + 7U), rank_roots[rank]);
  }
  if (!status.ok()) return fail_create(status);
  auto transaction_root = builder->finalize();
  if (!transaction_root.ok()) {
    return fail_create(transaction_root.status());
  }
  for (std::uint32_t rank = 0; rank < plan.world_size(); ++rank) {
    const auto& transfer_manifest = plan.rank_manifest(rank);
    auto metadata_receipt_root =
        compile_deepseek_rank_artifact_metadata_receipt_root(
            transfer_manifest, adoption_receipt_roots[rank],
            descriptor_transaction.transaction_root(), *transaction_root,
            payloads[rank].metadata_root, payloads[rank].blob_sha256,
            payloads[rank].blob.size(), plan.dspark_enabled());
    if (!metadata_receipt_root.ok()) {
      return fail_create(metadata_receipt_root.status());
    }
    auto mapping_owner_root =
        compile_deepseek_rank_artifact_mapping_owner_root(
            transfer_manifest, semantic_blobs[rank],
            *metadata_receipt_root,
            descriptor_transaction.transaction_root(), *transaction_root);
    if (!mapping_owner_root.ok()) {
      return fail_create(mapping_owner_root.status());
    }
    payloads[rank].expected_mapping_owner_root = *mapping_owner_root;
  }
  return DeepSeekRankArtifactMetadataTransferTransaction(
      std::move(descriptor_transaction), operations, std::move(payloads),
      std::move(*node_prefault_layout),
      *transaction_root);
}

Status DeepSeekRankArtifactMetadataTransferTransaction::fail(
    Status cause) noexcept {
  poisoned_ = true;
  complete_ = false;
  if (cause.ok()) {
    cause = Status::Internal(
        "DeepSeek artifact metadata transfer transaction failed");
  }
  const auto& fields = descriptor_transaction_.plan_.rank_manifest(0).fields();
  (void)operations_->abort_generation(
      fields.engine_epoch, fields.worker_generation, cause);
  return cause;
}

Status DeepSeekRankArtifactMetadataTransferTransaction::
    ensure_before_deadline() {
  auto now = operations_->monotonic_now_ns();
  if (!now.ok()) return now.status();
  const auto deadline =
      descriptor_transaction_.plan_.rank_manifest(0).fields().deadline_ns;
  if (*now >= deadline) {
    return Status::DeadlineExceeded(
        "DeepSeek artifact metadata transfer deadline expired");
  }
  return Status::Ok();
}

Result<DeepSeekRankArtifactMetadataChunk>
DeepSeekRankArtifactMetadataTransferTransaction::make_chunk(
    std::uint32_t rank, std::uint32_t chunk_index) const {
  if (rank >= payloads_.size() ||
      chunk_index >= payloads_[rank].chunk_count) {
    return Status::InvalidArgument(
        "DeepSeek artifact metadata chunk index is invalid");
  }
  const auto& payload = payloads_[rank];
  const auto& transfer =
      descriptor_transaction_.plan_.rank_manifest(rank);
  const auto& transfer_fields = transfer.fields();
  const auto offset =
      static_cast<std::uint64_t>(chunk_index) *
      kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes;
  const auto bytes = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes,
          payload.blob.size() - offset));
  std::vector<std::byte> chunk_payload(
      payload.blob.begin() + static_cast<std::ptrdiff_t>(offset),
      payload.blob.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
  auto chunk_sha256 = sha256(chunk_payload);
  if (!chunk_sha256.ok()) return chunk_sha256.status();
  DeepSeekRankArtifactMetadataChunkFields fields{
      transfer_fields.engine_epoch,
      transfer_fields.worker_generation,
      transfer_fields.world_size,
      rank,
      chunk_index,
      payload.chunk_count,
      offset,
      bytes,
      payload.blob.size(),
      transfer_fields.process_manifest_identity,
      transfer_fields.process_identity,
      transfer_fields.pidfd_identity,
      transfer_fields.control_identity,
      transfer_fields.challenge_identity,
      transfer.manifest_root(),
      descriptor_transaction_.transaction_root(),
      payload.metadata_root,
      transaction_root_,
      payload.blob_sha256,
      *chunk_sha256};
  return DeepSeekRankArtifactMetadataChunk::Create(
      std::move(fields), std::move(chunk_payload));
}

Status DeepSeekRankArtifactMetadataTransferTransaction::validate_ack(
    std::uint32_t rank,
    const DeepSeekRankArtifactMetadataChunkAck& ack) const {
  if (rank >= ranks_.size() || !ranks_[rank].in_flight_chunk ||
      !ranks_[rank].frame_accepted) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata ACK has no accepted chunk");
  }
  const auto& expected = ranks_[rank].in_flight_chunk->fields();
  const auto& fields = ack.fields();
  if (fields.engine_epoch != expected.engine_epoch ||
      fields.worker_generation != expected.worker_generation ||
      fields.world_size != expected.world_size || fields.rank != rank ||
      fields.chunk_index != expected.chunk_index ||
      fields.chunk_count != expected.chunk_count ||
      fields.cumulative_payload_bytes !=
          expected.payload_offset + expected.payload_bytes ||
      fields.total_blob_bytes != expected.total_blob_bytes ||
      fields.process_manifest_identity != expected.process_manifest_identity ||
      fields.process_identity != expected.process_identity ||
      fields.pidfd_identity != expected.pidfd_identity ||
      fields.control_identity != expected.control_identity ||
      fields.challenge_identity != expected.challenge_identity ||
      fields.transfer_manifest_root != expected.transfer_manifest_root ||
      fields.descriptor_transfer_transaction_root !=
          expected.descriptor_transfer_transaction_root ||
      fields.metadata_root != expected.metadata_root ||
      fields.metadata_transaction_root !=
          expected.metadata_transaction_root ||
      fields.blob_sha256 != expected.blob_sha256 ||
      fields.chunk_sha256 != expected.chunk_sha256) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata ACK differs from in-flight chunk");
  }
  return Status::Ok();
}

Status DeepSeekRankArtifactMetadataTransferTransaction::advance() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact metadata transfer transaction is poisoned");
  }
  if (complete_) return Status::Ok();

  for (std::uint32_t rank = 0; rank < ranks_.size(); ++rank) {
    auto& state = ranks_[rank];
    if (!state.in_flight_chunk &&
        state.next_chunk_index < payloads_[rank].chunk_count) {
      auto chunk = make_chunk(rank, state.next_chunk_index);
      if (!chunk.ok()) return fail(chunk.status());
      auto frame = encode_deepseek_rank_artifact_metadata_chunk(*chunk);
      if (!frame.ok()) return fail(frame.status());
      state.in_flight_chunk.emplace(std::move(*chunk));
      state.in_flight_frame = std::move(*frame);
    }

    if (state.in_flight_chunk && !state.frame_accepted) {
      auto status = ensure_before_deadline();
      if (!status.ok()) return fail(status);
      status = operations_->send_chunk(rank, state.in_flight_frame);
      if (!status.ok()) {
        if (status.code() == StatusCode::kUnavailable) continue;
        return fail(status);
      }
      state.frame_accepted = true;
    }

    if (state.in_flight_chunk && state.frame_accepted) {
      auto status = ensure_before_deadline();
      if (!status.ok()) return fail(status);
      auto frame = operations_->poll_ack(rank);
      if (!frame.ok()) {
        if (frame.status().code() == StatusCode::kUnavailable) continue;
        return fail(frame.status());
      }
      if (!frame->has_value()) continue;
      status = ensure_before_deadline();
      if (!status.ok()) return fail(status);
      auto ack = decode_deepseek_rank_artifact_metadata_chunk_ack(**frame);
      if (!ack.ok()) return fail(ack.status());
      status = validate_ack(rank, *ack);
      if (!status.ok()) return fail(status);
      state.acknowledged_bytes =
          ack->fields().cumulative_payload_bytes;
      ++state.next_chunk_index;
      state.in_flight_chunk.reset();
      state.in_flight_frame.clear();
      state.frame_accepted = false;
    }
  }

  bool all_complete = true;
  for (std::uint32_t rank = 0; rank < ranks_.size(); ++rank) {
    const auto& state = ranks_[rank];
    if (state.in_flight_chunk ||
        state.next_chunk_index != payloads_[rank].chunk_count ||
        state.acknowledged_bytes != payloads_[rank].blob.size()) {
      all_complete = false;
      break;
    }
  }
  if (all_complete) {
    complete_ = true;
    return Status::Ok();
  }
  return Status::Unavailable(
      "DeepSeek artifact metadata transfer transaction is pending");
}

const Sha256Digest&
DeepSeekRankArtifactMetadataTransferTransaction::metadata_root(
    std::uint32_t rank) const {
  if (rank >= payloads_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank payload");
  }
  return payloads_[rank].metadata_root;
}

const Sha256Digest&
DeepSeekRankArtifactMetadataTransferTransaction::blob_sha256(
    std::uint32_t rank) const {
  if (rank >= payloads_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank payload");
  }
  return payloads_[rank].blob_sha256;
}

std::uint64_t DeepSeekRankArtifactMetadataTransferTransaction::blob_bytes(
    std::uint32_t rank) const {
  if (rank >= payloads_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank payload");
  }
  return payloads_[rank].blob.size();
}

std::uint32_t DeepSeekRankArtifactMetadataTransferTransaction::chunk_count(
    std::uint32_t rank) const {
  if (rank >= payloads_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank payload");
  }
  return payloads_[rank].chunk_count;
}

std::uint64_t
DeepSeekRankArtifactMetadataTransferTransaction::acknowledged_bytes(
    std::uint32_t rank) const {
  if (rank >= ranks_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank state");
  }
  return ranks_[rank].acknowledged_bytes;
}

const Sha256Digest&
DeepSeekRankArtifactMetadataTransferTransaction::expected_mapping_owner_root(
    std::uint32_t rank) const {
  if (rank >= payloads_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank payload");
  }
  return payloads_[rank].expected_mapping_owner_root;
}

std::uint64_t DeepSeekRankArtifactMetadataTransferTransaction::
    expected_mapped_interval_bytes(std::uint32_t rank) const {
  if (rank >= payloads_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank payload");
  }
  return payloads_[rank].expected_mapped_interval_bytes;
}

const DeepSeekRankArtifactPrefaultLayout&
DeepSeekRankArtifactMetadataTransferTransaction::expected_prefault_layout(
    std::uint32_t rank) const {
  if (rank >= payloads_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank payload");
  }
  return payloads_[rank].expected_prefault_layout;
}

ArtifactImmutabilityMode
DeepSeekRankArtifactMetadataTransferTransaction::expected_immutability_mode(
    std::uint32_t rank) const {
  if (rank >= payloads_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank payload");
  }
  return payloads_[rank].expected_immutability_mode;
}

bool DeepSeekRankArtifactMetadataTransferTransaction::
    expected_source_catalog_production_eligible(
        std::uint32_t rank) const {
  if (rank >= payloads_.size()) {
    throw std::out_of_range("DeepSeek artifact metadata rank payload");
  }
  return payloads_[rank].expected_source_catalog_production_eligible;
}

}  // namespace pih
