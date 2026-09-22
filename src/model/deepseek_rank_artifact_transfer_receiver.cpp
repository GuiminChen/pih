#include "pih/model/deepseek_rank_artifact_transfer_receiver.h"

#include <algorithm>
#include <utility>

namespace pih {
namespace {

bool local_identity_is_valid(
    const DeepSeekRankProcessManifest& manifest,
    const DeepSeekRankExecReady& ready) noexcept {
  const auto& receipt = ready.receipt;
  return manifest.engine_epoch != 0 && manifest.worker_generation != 0 &&
         manifest.world_size >= 1 && manifest.world_size <= 4 &&
         manifest.rank < manifest.world_size &&
         manifest.engine_epoch == receipt.engine_epoch &&
         manifest.worker_generation == receipt.worker_generation &&
         manifest.rank == receipt.rank &&
         manifest.physical_device_identity ==
             receipt.physical_device_identity &&
         manifest.process_manifest_identity ==
             receipt.process_manifest_identity &&
         manifest.physical_device_uuid_commitment ==
             receipt.physical_device_uuid_commitment &&
         receipt.physical_device_uuid_commitment != Sha256Digest{} &&
         manifest.startup_device_ordinal == receipt.startup_device_ordinal &&
         manifest.startup_deadline_ns == receipt.startup_deadline_ns &&
         receipt.process_identity != 0 && receipt.pidfd_identity != 0 &&
         receipt.control_identity != 0 && ready.challenge_identity != 0;
}

}  // namespace

DeepSeekRankArtifactTransferReceiver::
    DeepSeekRankArtifactTransferReceiver(
        DeepSeekRankProcessManifest local_manifest,
        DeepSeekRankExecReady exec_ready, std::int32_t control_fd,
        std::uint64_t manifest_wait_deadline_ns,
        DeepSeekRankScmRightsInFlightLedger& scm_rights_ledger,
        DeepSeekRankArtifactTransferReceiverOperations& operations) noexcept
    : local_manifest_(std::move(local_manifest)),
      exec_ready_(std::move(exec_ready)), control_fd_(control_fd),
      manifest_wait_deadline_ns_(manifest_wait_deadline_ns),
      scm_rights_ledger_(&scm_rights_ledger), operations_(&operations) {}

Result<DeepSeekRankArtifactTransferReceiver>
DeepSeekRankArtifactTransferReceiver::Create(
    DeepSeekRankProcessManifest local_manifest,
    DeepSeekRankExecReady exec_ready, std::int32_t control_fd,
    std::uint64_t manifest_wait_deadline_ns,
    DeepSeekRankScmRightsInFlightLedger& scm_rights_ledger,
    DeepSeekRankArtifactTransferReceiverOperations& operations) {
  if (!local_identity_is_valid(local_manifest, exec_ready) ||
      control_fd < 0 || manifest_wait_deadline_ns == 0 ||
      manifest_wait_deadline_ns <= local_manifest.startup_deadline_ns) {
    return Status::InvalidArgument(
        "DeepSeek artifact transfer receiver identity is invalid");
  }
  return DeepSeekRankArtifactTransferReceiver(
      std::move(local_manifest), std::move(exec_ready), control_fd,
      manifest_wait_deadline_ns, scm_rights_ledger, operations);
}

Status DeepSeekRankArtifactTransferReceiver::fail(Status cause) noexcept {
  poisoned_ = true;
  complete_ = false;
  if (cause.ok()) {
    return Status::Internal(
        "DeepSeek artifact transfer receiver failed");
  }
  return cause;
}

Status DeepSeekRankArtifactTransferReceiver::ensure_before_deadline() {
  auto now = operations_->monotonic_now_ns();
  if (!now.ok()) {
    return Status::Internal(
        "DeepSeek artifact transfer receiver clock failed");
  }
  const auto deadline = transfer_manifest_
                            ? transfer_manifest_->fields().deadline_ns
                            : manifest_wait_deadline_ns_;
  if (*now >= deadline) {
    return Status::DeadlineExceeded(
        "DeepSeek artifact transfer receiver deadline expired");
  }
  return Status::Ok();
}

Status DeepSeekRankArtifactTransferReceiver::validate_manifest(
    const DeepSeekRankArtifactTransferManifest& manifest) const {
  const auto& fields = manifest.fields();
  const auto& receipt = exec_ready_.receipt;
  if (fields.engine_epoch != local_manifest_.engine_epoch ||
      fields.worker_generation != local_manifest_.worker_generation ||
      fields.world_size != local_manifest_.world_size ||
      fields.rank != local_manifest_.rank ||
      fields.process_manifest_identity !=
          local_manifest_.process_manifest_identity ||
      fields.process_identity != receipt.process_identity ||
      fields.pidfd_identity != receipt.pidfd_identity ||
      fields.control_identity != receipt.control_identity ||
      fields.challenge_identity != exec_ready_.challenge_identity ||
      fields.deadline_ns <= local_manifest_.startup_deadline_ns ||
      fields.deadline_ns > manifest_wait_deadline_ns_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer manifest differs from worker identity");
  }
  return Status::Ok();
}

Status DeepSeekRankArtifactTransferReceiver::accept_batch(
    DeepSeekRankArtifactReceivedBatch batch) {
  const auto& manifest_fields = transfer_manifest_->fields();
  const auto& fields = batch.metadata.fields();
  const auto expectations = batch.metadata.expectations();
  const auto expected_descriptor_count = std::min<std::uint32_t>(
      kDeepSeekRankArtifactTransferDescriptorBatchMaximum,
      manifest_fields.descriptor_count -
          next_batch_index_ *
              kDeepSeekRankArtifactTransferDescriptorBatchMaximum);
  if (pending_ack_ || in_flight_lease_ ||
      fields.engine_epoch != manifest_fields.engine_epoch ||
      fields.worker_generation != manifest_fields.worker_generation ||
      fields.world_size != manifest_fields.world_size ||
      fields.rank != manifest_fields.rank ||
      fields.batch_index != next_batch_index_ ||
      fields.batch_count != manifest_fields.descriptor_batch_count ||
      fields.first_descriptor_ordinal != adopted_expectations_.size() ||
      fields.descriptor_count != expected_descriptor_count ||
      fields.descriptor_count != expectations.size() ||
      fields.cumulative_descriptor_count !=
          adopted_expectations_.size() + expectations.size() ||
      fields.final_batch !=
          (next_batch_index_ + 1U ==
           manifest_fields.descriptor_batch_count) ||
      fields.process_manifest_identity !=
          manifest_fields.process_manifest_identity ||
      fields.process_identity != manifest_fields.process_identity ||
      fields.pidfd_identity != manifest_fields.pidfd_identity ||
      fields.control_identity != manifest_fields.control_identity ||
      fields.challenge_identity != manifest_fields.challenge_identity ||
      fields.transfer_manifest_root != transfer_manifest_->manifest_root() ||
      fields.artifact_admission_binding_root !=
          manifest_fields.artifact_admission_binding_root ||
      fields.cumulative_descriptor_count > manifest_fields.descriptor_count ||
      (fields.final_batch &&
       fields.cumulative_descriptor_count !=
           manifest_fields.descriptor_count) ||
      batch.descriptors.size() != expectations.size()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact descriptor batch differs from transfer manifest");
  }
  if (transaction_root_ &&
      *transaction_root_ != fields.transfer_transaction_root) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer transaction root changed");
  }

  for (std::size_t index = 0; index < expectations.size(); ++index) {
    if (expectations[index].ordinal !=
            adopted_expectations_.size() + index ||
        expectations[index].immutability_mode !=
            manifest_fields.immutability_mode ||
        batch.descriptors[index].shard_name !=
            expectations[index].shard_name ||
        batch.descriptors[index].descriptor.identity() !=
            expectations[index].identity) {
      return Status::FailedPrecondition(
          "DeepSeek received artifact descriptor identity differs");
    }
  }

  std::vector<DeepSeekRankArtifactDescriptorExpectation> candidate =
      adopted_expectations_;
  candidate.insert(candidate.end(), expectations.begin(), expectations.end());
  auto adopted_root =
      compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
          fields.rank, candidate);
  if (!adopted_root.ok()) return adopted_root.status();
  if (*adopted_root != fields.adopted_descriptor_set_root) {
    return Status::FailedPrecondition(
        "DeepSeek received artifact cumulative descriptor root differs");
  }
  auto lease = scm_rights_ledger_->acquire(fields.descriptor_count);
  if (!lease.ok()) return lease.status();

  if (!transaction_root_) transaction_root_ = fields.transfer_transaction_root;
  adopted_expectations_ = std::move(candidate);
  for (auto& descriptor : batch.descriptors) {
    adopted_descriptors_.push_back(std::move(descriptor));
  }
  DeepSeekRankArtifactTransferAckFields ack_fields{
      fields.engine_epoch,
      fields.worker_generation,
      fields.world_size,
      fields.rank,
      fields.batch_index,
      fields.batch_count,
      fields.first_descriptor_ordinal,
      fields.descriptor_count,
      fields.cumulative_descriptor_count,
      fields.final_batch,
      fields.process_manifest_identity,
      fields.process_identity,
      fields.pidfd_identity,
      fields.control_identity,
      fields.challenge_identity,
      fields.transfer_manifest_root,
      fields.artifact_admission_binding_root,
      fields.transfer_transaction_root,
      fields.descriptor_batch_root,
      fields.adopted_descriptor_set_root};
  auto ack = DeepSeekRankArtifactTransferAck::Create(ack_fields);
  if (!ack.ok()) return ack.status();
  pending_ack_ = encode_deepseek_rank_artifact_transfer_ack(*ack);
  in_flight_lease_.emplace(std::move(*lease));
  pending_ack_is_final_ = fields.final_batch;
  return Status::Ok();
}

Status DeepSeekRankArtifactTransferReceiver::flush_ack() {
  if (!pending_ack_ || !in_flight_lease_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer receiver ACK state is incomplete");
  }
  auto status = ensure_before_deadline();
  if (!status.ok()) return status;
  status = operations_->send_ack(control_fd_, *pending_ack_);
  if (!status.ok()) return status;
  status = ensure_before_deadline();
  if (!status.ok()) return status;
  pending_ack_.reset();
  in_flight_lease_.reset();
  ++next_batch_index_;
  if (pending_ack_is_final_) {
    if (adopted_descriptors_.size() !=
            transfer_manifest_->fields().descriptor_count ||
        next_batch_index_ !=
            transfer_manifest_->fields().descriptor_batch_count) {
      return Status::FailedPrecondition(
          "DeepSeek artifact transfer receiver final inventory differs");
    }
    complete_ = true;
  }
  pending_ack_is_final_ = false;
  return Status::Ok();
}

Status DeepSeekRankArtifactTransferReceiver::advance() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer receiver is poisoned");
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
  if (!transfer_manifest_) {
    auto frame = operations_->receive_manifest(control_fd_);
    if (!frame.ok()) {
      if (frame.status().code() == StatusCode::kUnavailable) {
        return frame.status();
      }
      return fail(frame.status());
    }
    if (!frame->has_value()) {
      return Status::Unavailable(
          "DeepSeek artifact transfer manifest is pending");
    }
    status = ensure_before_deadline();
    if (!status.ok()) return fail(status);
    auto manifest = decode_deepseek_rank_artifact_transfer_manifest(**frame);
    if (!manifest.ok()) return fail(manifest.status());
    status = validate_manifest(*manifest);
    if (!status.ok()) return fail(status);
    transfer_manifest_ = std::move(*manifest);
  }

  status = ensure_before_deadline();
  if (!status.ok()) return fail(status);
  auto batch = operations_->receive_descriptor_batch(control_fd_);
  if (!batch.ok()) {
    if (batch.status().code() == StatusCode::kUnavailable) {
      return batch.status();
    }
    return fail(batch.status());
  }
  if (!batch->has_value()) {
    return Status::Unavailable(
        "DeepSeek artifact descriptor batch is pending");
  }
  status = ensure_before_deadline();
  if (!status.ok()) return fail(status);
  status = accept_batch(std::move(**batch));
  if (!status.ok()) return fail(status);
  status = flush_ack();
  if (!status.ok()) {
    if (status.code() == StatusCode::kUnavailable) return status;
    return fail(status);
  }
  return complete_ ? Status::Ok()
                   : Status::Unavailable(
                         "DeepSeek artifact transfer receiver is pending");
}

}  // namespace pih
