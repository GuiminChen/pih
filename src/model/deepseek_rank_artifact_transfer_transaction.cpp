#include "pih/model/deepseek_rank_artifact_transfer_transaction.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

Result<Sha256Digest> compile_descriptor_root(
    const DeepSeekRankArtifactDescriptorExpectation& descriptor) {
  const auto digest_present = nonzero(descriptor.enforced_digest);
  if (descriptor.shard_name.empty() || descriptor.ordinal >=
          kDeepSeekRankArtifactTransferDescriptorMaximum ||
      descriptor.identity.file_bytes == 0 ||
      descriptor.identity.filesystem_identity == 0 ||
      descriptor.identity.file_identity == 0 ||
      descriptor.identity.data_mtime_seconds < 0 ||
      descriptor.identity.data_mtime_nanoseconds >= 1'000'000'000U ||
      (descriptor.immutability_mode == ArtifactImmutabilityMode::kFsVerity) !=
          digest_present ||
      descriptor.immutability_mode ==
          ArtifactImmutabilityMode::kDmVeritySnapshot) {
    return Status::InvalidArgument(
        "DeepSeek artifact transfer descriptor identity is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-transfer-descriptor:v1",
      digest_present ? 10U : 9U);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, descriptor.ordinal);
  if (status.ok()) status = builder->add_ascii_utf8(2, descriptor.shard_name);
  if (status.ok()) {
    status = builder->add_u64(3, descriptor.identity.file_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(4, descriptor.identity.filesystem_identity);
  }
  if (status.ok()) {
    status = builder->add_u64(5, descriptor.identity.file_identity);
  }
  if (status.ok()) {
    status = builder->add_u64(
        6, static_cast<std::uint64_t>(
               descriptor.identity.data_mtime_seconds));
  }
  if (status.ok()) {
    status = builder->add_u32(
        7, descriptor.identity.data_mtime_nanoseconds);
  }
  if (status.ok()) {
    status = builder->add_u32(
        8, static_cast<std::uint32_t>(descriptor.immutability_mode));
  }
  if (status.ok()) status = builder->add_u32(9, digest_present ? 1U : 0U);
  if (status.ok() && digest_present) {
    status = builder->add_hash(10, descriptor.enforced_digest);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> compile_batch_root(
    std::uint32_t rank, std::uint32_t batch_index,
    std::span<const Sha256Digest> descriptor_roots) {
  if (descriptor_roots.empty() ||
      descriptor_roots.size() >
          kDeepSeekRankArtifactTransferDescriptorBatchMaximum) {
    return Status::InvalidArgument(
        "DeepSeek artifact transfer descriptor batch is invalid");
  }
  const auto first = batch_index *
                     kDeepSeekRankArtifactTransferDescriptorBatchMaximum;
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-transfer-descriptor-batch:v1",
      static_cast<std::uint32_t>(descriptor_roots.size() + 4U));
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, rank);
  if (status.ok()) status = builder->add_u32(2, batch_index);
  if (status.ok()) status = builder->add_u32(3, first);
  if (status.ok()) {
    status = builder->add_u32(
        4, static_cast<std::uint32_t>(descriptor_roots.size()));
  }
  for (std::size_t index = 0;
       status.ok() && index < descriptor_roots.size(); ++index) {
    status = builder->add_hash(
        static_cast<std::uint16_t>(index + 5U), descriptor_roots[index]);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> compile_adopted_root(
    std::uint32_t rank,
    std::span<const Sha256Digest> descriptor_roots) {
  if (descriptor_roots.empty() || descriptor_roots.size() >
          kDeepSeekRankArtifactTransferDescriptorMaximum) {
    return Status::InvalidArgument(
        "DeepSeek adopted descriptor set is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-transfer-adopted-set:v1",
      static_cast<std::uint32_t>(descriptor_roots.size() + 2U));
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, rank);
  if (status.ok()) {
    status = builder->add_u32(
        2, static_cast<std::uint32_t>(descriptor_roots.size()));
  }
  for (std::size_t index = 0;
       status.ok() && index < descriptor_roots.size(); ++index) {
    status = builder->add_hash(
        static_cast<std::uint16_t>(index + 3U), descriptor_roots[index]);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

}  // namespace

Result<Sha256Digest>
compile_deepseek_rank_artifact_transfer_descriptor_root(
    const DeepSeekRankArtifactDescriptorExpectation& descriptor) {
  return compile_descriptor_root(descriptor);
}

Result<Sha256Digest>
compile_deepseek_rank_artifact_transfer_descriptor_batch_root(
    std::uint32_t rank, std::uint32_t batch_index,
    std::span<const DeepSeekRankArtifactDescriptorExpectation> descriptors) {
  const auto expected_first =
      batch_index * kDeepSeekRankArtifactTransferDescriptorBatchMaximum;
  if (descriptors.empty() ||
      descriptors.size() >
          kDeepSeekRankArtifactTransferDescriptorBatchMaximum) {
    return Status::InvalidArgument(
        "DeepSeek artifact transfer descriptor batch is invalid");
  }
  std::vector<Sha256Digest> roots;
  roots.reserve(descriptors.size());
  for (std::size_t index = 0; index < descriptors.size(); ++index) {
    if (descriptors[index].ordinal != expected_first + index) {
      return Status::InvalidArgument(
          "DeepSeek artifact transfer descriptor batch order is invalid");
    }
    auto root = compile_descriptor_root(descriptors[index]);
    if (!root.ok()) return root.status();
    roots.push_back(*root);
  }
  return compile_batch_root(rank, batch_index, roots);
}

Result<Sha256Digest>
compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
    std::uint32_t rank,
    std::span<const DeepSeekRankArtifactDescriptorExpectation> descriptors) {
  if (descriptors.empty() || descriptors.size() >
          kDeepSeekRankArtifactTransferDescriptorMaximum) {
    return Status::InvalidArgument(
        "DeepSeek adopted descriptor set is invalid");
  }
  std::vector<Sha256Digest> roots;
  roots.reserve(descriptors.size());
  for (std::size_t index = 0; index < descriptors.size(); ++index) {
    if (descriptors[index].ordinal != index) {
      return Status::InvalidArgument(
          "DeepSeek adopted descriptor set order is invalid");
    }
    auto root = compile_descriptor_root(descriptors[index]);
    if (!root.ok()) return root.status();
    roots.push_back(*root);
  }
  return compile_adopted_root(rank, roots);
}

DeepSeekRankArtifactTransferTransaction::
    DeepSeekRankArtifactTransferTransaction(
        DeepSeekRankArtifactTransferPlan plan,
        DeepSeekRankArtifactTransferOperations& operations,
        std::vector<std::array<
            std::byte, kDeepSeekRankArtifactTransferManifestBytes>>
            manifest_frames,
        std::vector<std::vector<BatchState>> batches,
        Sha256Digest transaction_root) noexcept
    : plan_(std::move(plan)), operations_(&operations),
      manifest_frames_(std::move(manifest_frames)),
      batches_(std::move(batches)), ranks_(plan_.world_size()),
      transaction_root_(transaction_root) {}

Result<DeepSeekRankArtifactTransferTransaction>
DeepSeekRankArtifactTransferTransaction::Create(
    DeepSeekRankArtifactTransferPlan plan,
    DeepSeekRankArtifactTransferOperations& operations) {
  if (!plan.owns_all_antecedents() || plan.world_size() < 1 ||
      plan.world_size() > 4 || !nonzero(plan.plan_root()) ||
      !nonzero(plan.artifact_root()) ||
      !nonzero(plan.artifact_binding_root())) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer transaction lacks owned authority");
  }

  std::vector<std::array<
      std::byte, kDeepSeekRankArtifactTransferManifestBytes>>
      manifest_frames;
  std::vector<std::vector<BatchState>> batches;
  std::vector<Sha256Digest> rank_roots;
  manifest_frames.reserve(plan.world_size());
  batches.reserve(plan.world_size());
  rank_roots.reserve(plan.world_size());
  const auto deadline = plan.rank_manifest(0).fields().deadline_ns;

  for (std::uint32_t rank = 0; rank < plan.world_size(); ++rank) {
    const auto& transfer_manifest = plan.rank_manifest(rank);
    const auto& fields = transfer_manifest.fields();
    const auto& handoff_manifest = plan.handoff_.rank_manifest(rank);
    const auto expectations = handoff_manifest.descriptor_expectations();
    const auto descriptors = plan.handoff_.descriptors(rank);
    if (fields.rank != rank || fields.world_size != plan.world_size() ||
        fields.deadline_ns != deadline ||
        fields.artifact_admission_binding_root !=
            plan.artifact_binding_root() ||
        fields.descriptor_count != expectations.size() ||
        descriptors.size() != expectations.size()) {
      return Status::FailedPrecondition(
          "DeepSeek artifact transfer rank inventory drifted");
    }

    std::vector<Sha256Digest> descriptor_roots;
    descriptor_roots.reserve(expectations.size());
    for (std::size_t index = 0; index < expectations.size(); ++index) {
      if (expectations[index].ordinal != index ||
          descriptors[index].shard_name != expectations[index].shard_name ||
          descriptors[index].descriptor.identity() !=
              expectations[index].identity) {
        return Status::FailedPrecondition(
            "DeepSeek artifact transfer descriptor inventory drifted");
      }
      auto root = compile_descriptor_root(expectations[index]);
      if (!root.ok()) return root.status();
      descriptor_roots.push_back(*root);
    }

    std::vector<BatchState> rank_batches;
    rank_batches.reserve(fields.descriptor_batch_count);
    for (std::uint32_t batch_index = 0;
         batch_index < fields.descriptor_batch_count; ++batch_index) {
      const auto first =
          batch_index * kDeepSeekRankArtifactTransferDescriptorBatchMaximum;
      const auto count = std::min<std::uint32_t>(
          kDeepSeekRankArtifactTransferDescriptorBatchMaximum,
          fields.descriptor_count - first);
      auto batch_root = compile_batch_root(
          rank, batch_index,
          std::span<const Sha256Digest>(descriptor_roots).subspan(first,
                                                                  count));
      if (!batch_root.ok()) return batch_root.status();
      auto adopted_root = compile_adopted_root(
          rank, std::span<const Sha256Digest>(descriptor_roots).first(
                    first + count));
      if (!adopted_root.ok()) return adopted_root.status();
      rank_batches.push_back(
          {first, count, *batch_root, *adopted_root});
    }
    if (rank_batches.size() != fields.descriptor_batch_count ||
        rank_batches.empty()) {
      return Status::FailedPrecondition(
          "DeepSeek artifact transfer batch geometry drifted");
    }

    auto rank_builder = CanonicalHashBuilder::Create(
        "pih:deepseek-rank-artifact-transfer-transaction-rank:v1",
        static_cast<std::uint32_t>(rank_batches.size() + 3U));
    if (!rank_builder.ok()) return rank_builder.status();
    auto status = rank_builder->add_hash(
        1, transfer_manifest.manifest_root());
    if (status.ok()) {
      status = rank_builder->add_u32(
          2, static_cast<std::uint32_t>(rank_batches.size()));
    }
    if (status.ok()) {
      status = rank_builder->add_hash(
          3, rank_batches.back().adopted_descriptor_set_root);
    }
    for (std::size_t index = 0;
         status.ok() && index < rank_batches.size(); ++index) {
      status = rank_builder->add_hash(
          static_cast<std::uint16_t>(index + 4U),
          rank_batches[index].descriptor_batch_root);
    }
    if (!status.ok()) return status;
    auto rank_root = rank_builder->finalize();
    if (!rank_root.ok()) return rank_root.status();
    rank_roots.push_back(*rank_root);
    manifest_frames.push_back(plan.encode_rank_manifest(rank));
    batches.push_back(std::move(rank_batches));
  }

  auto transaction_builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-transfer-transaction:v1",
      plan.world_size() + 5U);
  if (!transaction_builder.ok()) return transaction_builder.status();
  auto status = transaction_builder->add_hash(1, plan.plan_root());
  if (status.ok()) status = transaction_builder->add_u32(2, plan.world_size());
  if (status.ok()) status = transaction_builder->add_u64(3, deadline);
  if (status.ok()) status = transaction_builder->add_hash(4, plan.artifact_root());
  if (status.ok()) {
    status = transaction_builder->add_hash(5, plan.artifact_binding_root());
  }
  for (std::uint32_t rank = 0;
       status.ok() && rank < plan.world_size(); ++rank) {
    status = transaction_builder->add_hash(
        static_cast<std::uint16_t>(rank + 6U), rank_roots[rank]);
  }
  if (!status.ok()) return status;
  auto transaction_root = transaction_builder->finalize();
  if (!transaction_root.ok()) return transaction_root.status();
  return DeepSeekRankArtifactTransferTransaction(
      std::move(plan), operations, std::move(manifest_frames),
      std::move(batches), *transaction_root);
}

Status DeepSeekRankArtifactTransferTransaction::fail(Status cause) noexcept {
  poisoned_ = true;
  complete_ = false;
  if (cause.ok()) {
    cause = Status::Internal(
        "DeepSeek artifact transfer transaction failed");
  }
  (void)operations_->abort_generation(
      plan_.rank_manifest(0).fields().engine_epoch,
      plan_.rank_manifest(0).fields().worker_generation, cause);
  return cause;
}

Status DeepSeekRankArtifactTransferTransaction::ensure_before_deadline() {
  auto now = operations_->monotonic_now_ns();
  if (!now.ok()) return now.status();
  if (*now >= plan_.rank_manifest(0).fields().deadline_ns) {
    return Status::DeadlineExceeded(
        "DeepSeek artifact transfer transaction deadline expired");
  }
  return Status::Ok();
}

DeepSeekRankArtifactDescriptorBatchView
DeepSeekRankArtifactTransferTransaction::batch_view(
    std::uint32_t rank, std::uint32_t batch_index) const {
  const auto& manifest = plan_.rank_manifest(rank);
  const auto& fields = manifest.fields();
  const auto& batch = batches_[rank][batch_index];
  const auto expectations =
      plan_.handoff_.rank_manifest(rank).descriptor_expectations().subspan(
          batch.first_descriptor_ordinal, batch.descriptor_count);
  const auto descriptors = plan_.handoff_.descriptors(rank).subspan(
      batch.first_descriptor_ordinal, batch.descriptor_count);
  return {fields.engine_epoch,
          fields.worker_generation,
          fields.world_size,
          rank,
          batch_index,
          fields.descriptor_batch_count,
          batch.first_descriptor_ordinal,
          batch.first_descriptor_ordinal + batch.descriptor_count,
          batch_index + 1U == fields.descriptor_batch_count,
          fields.process_manifest_identity,
          fields.process_identity,
          fields.pidfd_identity,
          fields.control_identity,
          fields.challenge_identity,
          manifest.manifest_root(),
          fields.artifact_admission_binding_root,
          transaction_root_,
          batch.descriptor_batch_root,
          batch.adopted_descriptor_set_root,
          expectations,
          descriptors};
}

Status DeepSeekRankArtifactTransferTransaction::validate_ack(
    std::uint32_t rank,
    const DeepSeekRankArtifactTransferAck& ack) const {
  const auto& state = ranks_[rank];
  if (!state.in_flight_batch_index) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer acknowledgement has no in-flight batch");
  }
  const auto batch_index = *state.in_flight_batch_index;
  const auto expected = batch_view(rank, batch_index);
  const auto& fields = ack.fields();
  if (fields.engine_epoch != expected.engine_epoch ||
      fields.worker_generation != expected.worker_generation ||
      fields.world_size != expected.world_size || fields.rank != rank ||
      fields.batch_index != batch_index ||
      fields.batch_count != expected.batch_count ||
      fields.first_descriptor_ordinal != expected.first_descriptor_ordinal ||
      fields.descriptor_count != expected.descriptors.size() ||
      fields.cumulative_descriptor_count !=
          expected.cumulative_descriptor_count ||
      fields.final_batch != expected.final_batch ||
      fields.process_manifest_identity !=
          expected.process_manifest_identity ||
      fields.process_identity != expected.process_identity ||
      fields.pidfd_identity != expected.pidfd_identity ||
      fields.control_identity != expected.control_identity ||
      fields.challenge_identity != expected.challenge_identity ||
      fields.transfer_manifest_root != expected.transfer_manifest_root ||
      fields.artifact_admission_binding_root !=
          expected.artifact_admission_binding_root ||
      fields.transfer_transaction_root != expected.transfer_transaction_root ||
      fields.descriptor_batch_root != expected.descriptor_batch_root ||
      fields.adopted_descriptor_set_root !=
          expected.adopted_descriptor_set_root) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer acknowledgement differs from in-flight batch");
  }
  return Status::Ok();
}

Status DeepSeekRankArtifactTransferTransaction::advance() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer transaction is poisoned");
  }
  if (complete_) return Status::Ok();

  for (std::uint32_t rank = 0; rank < plan_.world_size(); ++rank) {
    auto& state = ranks_[rank];
    if (!state.manifest_accepted) {
      auto status = ensure_before_deadline();
      if (!status.ok()) return fail(status);
      status = operations_->send_manifest(rank, manifest_frames_[rank]);
      if (!status.ok()) {
        if (status.code() == StatusCode::kUnavailable) continue;
        return fail(status);
      }
      state.manifest_accepted = true;
    }

    if (state.in_flight_batch_index) {
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
      auto ack = decode_deepseek_rank_artifact_transfer_ack(**frame);
      if (!ack.ok()) return fail(ack.status());
      status = validate_ack(rank, *ack);
      if (!status.ok()) return fail(status);
      const auto& batch =
          batches_[rank][*state.in_flight_batch_index];
      state.acknowledged_descriptor_count =
          batch.first_descriptor_ordinal + batch.descriptor_count;
      ++state.next_batch_index;
      state.in_flight_batch_index.reset();
    }

    if (!state.in_flight_batch_index &&
        state.next_batch_index < batches_[rank].size()) {
      auto status = ensure_before_deadline();
      if (!status.ok()) return fail(status);
      const auto view = batch_view(rank, state.next_batch_index);
      status = operations_->send_descriptor_batch(view);
      if (!status.ok()) {
        if (status.code() == StatusCode::kUnavailable) continue;
        return fail(status);
      }
      state.in_flight_batch_index = state.next_batch_index;
    }
  }

  bool all_complete = true;
  for (std::uint32_t rank = 0; rank < plan_.world_size(); ++rank) {
    const auto& state = ranks_[rank];
    const auto descriptor_count =
        plan_.rank_manifest(rank).fields().descriptor_count;
    if (!state.manifest_accepted || state.in_flight_batch_index ||
        state.next_batch_index != batches_[rank].size() ||
        state.acknowledged_descriptor_count != descriptor_count) {
      all_complete = false;
      break;
    }
  }
  if (all_complete) {
    complete_ = true;
    return Status::Ok();
  }
  return Status::Unavailable(
      "DeepSeek artifact transfer transaction is pending");
}

bool DeepSeekRankArtifactTransferTransaction::manifest_accepted(
    std::uint32_t rank) const {
  if (rank >= ranks_.size()) {
    throw std::out_of_range("DeepSeek artifact transfer rank state");
  }
  return ranks_[rank].manifest_accepted;
}

std::uint32_t
DeepSeekRankArtifactTransferTransaction::acknowledged_descriptor_count(
    std::uint32_t rank) const {
  if (rank >= ranks_.size()) {
    throw std::out_of_range("DeepSeek artifact transfer rank state");
  }
  return ranks_[rank].acknowledged_descriptor_count;
}

}  // namespace pih
