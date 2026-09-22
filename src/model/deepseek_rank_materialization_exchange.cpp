#include "pih/model/deepseek_rank_materialization_exchange.h"

#include <utility>

namespace pih {
namespace {

bool local_identity_matches_report(
    const DeepSeekRankProcessManifest& manifest,
    const DeepSeekRankExecReady& ready,
    const DeepSeekRankArtifactMappingOwner& mapping_owner,
    const DeepSeekRankPostMappingResourceReport& report) noexcept {
  const auto& receipt = ready.receipt;
  return manifest.engine_epoch != 0 && manifest.worker_generation != 0 &&
         manifest.world_size >= 1 && manifest.world_size <= 4 &&
         manifest.rank < manifest.world_size &&
         receipt.engine_epoch == manifest.engine_epoch &&
         receipt.worker_generation == manifest.worker_generation &&
         receipt.rank == manifest.rank &&
         receipt.process_manifest_identity ==
             manifest.process_manifest_identity &&
         receipt.process_identity != 0 && receipt.pidfd_identity != 0 &&
         receipt.control_identity != 0 && ready.challenge_identity != 0 &&
         mapping_owner.engine_epoch() == manifest.engine_epoch &&
         mapping_owner.worker_generation() == manifest.worker_generation &&
         mapping_owner.world_size() == manifest.world_size &&
         mapping_owner.rank() == manifest.rank &&
         report.engine_epoch() == manifest.engine_epoch &&
         report.worker_generation() == manifest.worker_generation &&
         report.world_size() == manifest.world_size &&
         report.rank() == manifest.rank &&
         report.process_identity() == receipt.process_identity &&
         report.mapping_owner_root() == mapping_owner.mapping_owner_root() &&
         report.authority_deadline_ns() != 0 &&
         report.report_root() != Sha256Digest{};
}

}  // namespace

DeepSeekRankMaterializationGrantReceiver::
    DeepSeekRankMaterializationGrantReceiver(
        DeepSeekRankProcessManifest manifest,
        DeepSeekRankExecReady exec_ready, std::int32_t control_fd,
        std::uint64_t grant_wait_deadline_ns,
        const DeepSeekRankArtifactMappingOwner& mapping_owner,
        const DeepSeekRankPostMappingResourceReport& report,
        DeepSeekRankMaterializationGrantReceiverOperations& operations)
        noexcept
    : manifest_(std::move(manifest)), exec_ready_(std::move(exec_ready)),
      control_fd_(control_fd),
      grant_wait_deadline_ns_(grant_wait_deadline_ns),
      mapping_owner_(&mapping_owner), report_(&report),
      operations_(&operations) {}

Result<DeepSeekRankMaterializationGrantReceiver>
DeepSeekRankMaterializationGrantReceiver::Create(
    DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
    std::int32_t control_fd, std::uint64_t grant_wait_deadline_ns,
    const DeepSeekRankArtifactMappingOwner& mapping_owner,
    const DeepSeekRankPostMappingResourceReport& report,
    DeepSeekRankMaterializationGrantReceiverOperations& operations) {
  if (control_fd < 0 ||
      !local_identity_matches_report(
          manifest, exec_ready, mapping_owner, report) ||
      grant_wait_deadline_ns <= report.authority_deadline_ns()) {
    return Status::InvalidArgument(
        "DeepSeek materialization grant receiver identity is invalid");
  }
  return DeepSeekRankMaterializationGrantReceiver(
      std::move(manifest), std::move(exec_ready), control_fd,
      grant_wait_deadline_ns, mapping_owner, report, operations);
}

Status DeepSeekRankMaterializationGrantReceiver::fail(
    Status cause) noexcept {
  poisoned_ = true;
  complete_ = false;
  admission_.reset();
  ack_frame_.reset();
  return cause.ok()
             ? Status::Internal(
                   "DeepSeek materialization grant receiver failed")
             : cause;
}

std::uint64_t
DeepSeekRankMaterializationGrantReceiver::wait_deadline_ns()
    const noexcept {
  return admission_ ? admission_->deadline_ns()
                    : grant_wait_deadline_ns_;
}

Status DeepSeekRankMaterializationGrantReceiver::
    ensure_before_deadline() {
  auto now = operations_->monotonic_now_ns();
  if (!now.ok()) return now.status();
  if (*now >= wait_deadline_ns()) {
    return Status::DeadlineExceeded(
        "DeepSeek materialization grant receiver deadline expired");
  }
  return Status::Ok();
}

Status DeepSeekRankMaterializationGrantReceiver::flush_ack() {
  if (!admission_ || !ack_frame_) {
    return Status::FailedPrecondition(
        "DeepSeek materialization grant ACK state is incomplete");
  }
  auto status = ensure_before_deadline();
  if (!status.ok()) return status;
  status = operations_->send_ack(control_fd_, *ack_frame_);
  if (!status.ok()) return status;
  status = ensure_before_deadline();
  if (!status.ok()) return status;
  ack_frame_.reset();
  complete_ = true;
  return Status::Ok();
}

Status DeepSeekRankMaterializationGrantReceiver::advance() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek materialization grant receiver is poisoned");
  }
  if (complete_) return Status::Ok();
  if (ack_frame_) {
    auto status = flush_ack();
    if (!status.ok()) {
      if (status.code() == StatusCode::kUnavailable) return status;
      return fail(status);
    }
    return Status::Ok();
  }

  auto status = ensure_before_deadline();
  if (!status.ok()) return fail(status);
  auto frame = operations_->receive_grant(control_fd_);
  if (!frame.ok()) {
    if (frame.status().code() == StatusCode::kUnavailable) {
      return frame.status();
    }
    return fail(frame.status());
  }
  if (!frame->has_value()) {
    return Status::Unavailable(
        "DeepSeek materialization grant is pending");
  }
  status = ensure_before_deadline();
  if (!status.ok()) return fail(status);
  auto grant = decode_deepseek_rank_materialization_grant(**frame);
  if (!grant.ok()) return fail(grant.status());
  if (grant->deadline_ns > grant_wait_deadline_ns_ ||
      grant->deadline_ns <= report_->authority_deadline_ns()) {
    return fail(Status::FailedPrecondition(
        "DeepSeek materialization grant deadline differs"));
  }
  auto admission = DeepSeekRankMaterializationAdmission::Accept(
      manifest_, exec_ready_, *mapping_owner_, *report_,
      std::move(*grant));
  if (!admission.ok()) return fail(admission.status());
  auto ack = compile_deepseek_rank_materialization_grant_ack(*admission);
  if (!ack.ok()) return fail(ack.status());
  auto encoded = encode_deepseek_rank_materialization_grant_ack(*ack);
  if (!encoded.ok()) return fail(encoded.status());
  admission_ = std::move(*admission);
  ack_frame_ = std::move(*encoded);
  status = flush_ack();
  if (!status.ok()) {
    if (status.code() == StatusCode::kUnavailable) return status;
    return fail(status);
  }
  return Status::Ok();
}

std::optional<DeepSeekRankMaterializationGrantReceiverWaitEvent>
DeepSeekRankMaterializationGrantReceiver::wait_event() const noexcept {
  if (complete_ || poisoned_) return std::nullopt;
  return ack_frame_
             ? DeepSeekRankMaterializationGrantReceiverWaitEvent::
                   kAckWritable
             : DeepSeekRankMaterializationGrantReceiverWaitEvent::
                   kGrantReadable;
}

Result<DeepSeekRankMaterializationAdmission>
DeepSeekRankMaterializationGrantReceiver::take_admission() {
  if (!complete_ || !admission_) {
    return Status::FailedPrecondition(
        "DeepSeek materialization admission is not available");
  }
  auto result = std::move(*admission_);
  admission_.reset();
  return result;
}

DeepSeekRankMaterializationGrantCoordinator::
    DeepSeekRankMaterializationGrantCoordinator(
        const RuntimeEngineAdmission& admission,
        DeepSeekRankProcessSupervisor& supervisor,
        std::vector<DeepSeekRankProcessManifest> manifests,
        const DeepSeekRankPostMappingResourceCoordinator& post_mapping,
        std::vector<DeepSeekRankMaterializationGrantFields> grants,
        std::vector<std::array<
            std::byte, kDeepSeekRankMaterializationGrantFrameBytes>> frames,
        std::uint64_t deadline_ns,
        DeepSeekRankMaterializationGrantChannel& channel) noexcept
    : admission_(&admission), supervisor_(&supervisor),
      manifests_(std::move(manifests)), post_mapping_(&post_mapping),
      grants_(std::move(grants)), frames_(std::move(frames)),
      grants_dispatched_(manifests_.size(), false),
      acknowledgments_(manifests_.size()), deadline_ns_(deadline_ns),
      channel_(&channel) {}

Result<DeepSeekRankMaterializationGrantCoordinator>
DeepSeekRankMaterializationGrantCoordinator::Create(
    const RuntimeEngineAdmission& admission,
    DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankPostMappingResourceCoordinator& post_mapping,
    std::uint64_t deadline_ns,
    std::span<const DeepSeekRankMaterializationAllocationAuthority>
        allocation_authorities,
    DeepSeekRankMaterializationGrantChannel& channel) {
  const auto fail_create =
      [&supervisor](Status cause)
      -> Result<DeepSeekRankMaterializationGrantCoordinator> {
    if (cause.ok()) {
      cause = Status::Internal(
          "DeepSeek materialization grant coordinator construction failed");
    }
    if (supervisor.ready() && !supervisor.failed()) {
      (void)supervisor.abort_post_exec(cause);
    }
    return cause;
  };
  if (manifests.empty() || manifests.size() > 4 ||
      allocation_authorities.size() != manifests.size() ||
      !post_mapping.sealed() || post_mapping.poisoned() ||
      deadline_ns <= post_mapping.deadline_ns()) {
    return fail_create(Status::InvalidArgument(
        "DeepSeek materialization grant coordinator inputs are invalid"));
  }
  std::vector<DeepSeekRankMaterializationGrantFields> grants;
  std::vector<std::array<
      std::byte, kDeepSeekRankMaterializationGrantFrameBytes>> frames;
  grants.reserve(manifests.size());
  frames.reserve(manifests.size());
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    auto grant = compile_deepseek_rank_materialization_grant(
        admission, supervisor, manifests, post_mapping, rank, deadline_ns,
        allocation_authorities[rank]);
    if (!grant.ok()) return fail_create(grant.status());
    auto frame = encode_deepseek_rank_materialization_grant(*grant);
    if (!frame.ok()) return fail_create(frame.status());
    grants.push_back(std::move(*grant));
    frames.push_back(std::move(*frame));
  }
  return DeepSeekRankMaterializationGrantCoordinator(
      admission, supervisor,
      std::vector<DeepSeekRankProcessManifest>(
          manifests.begin(), manifests.end()),
      post_mapping, std::move(grants), std::move(frames), deadline_ns,
      channel);
}

Status DeepSeekRankMaterializationGrantCoordinator::fail(
    Status cause) noexcept {
  poisoned_ = true;
  complete_ = false;
  if (!supervisor_->failed()) {
    return supervisor_->abort_post_exec(
        cause.ok()
            ? Status::Internal(
                  "DeepSeek materialization grant coordination failed")
            : cause);
  }
  return cause.ok()
             ? Status::Internal(
                   "DeepSeek materialization grant coordination failed")
             : cause;
}

Status DeepSeekRankMaterializationGrantCoordinator::
    ensure_before_deadline() {
  auto now = channel_->monotonic_now_ns();
  if (!now.ok()) return now.status();
  if (*now >= deadline_ns_) {
    return Status::DeadlineExceeded(
        "DeepSeek materialization grant deadline expired");
  }
  return Status::Ok();
}

Status DeepSeekRankMaterializationGrantCoordinator::validate_ack(
    std::uint32_t rank,
    const DeepSeekRankMaterializationGrantAckFields& ack) const {
  if (rank >= grants_.size() || !grants_dispatched_[rank]) {
    return Status::FailedPrecondition(
        "DeepSeek materialization ACK has no accepted grant");
  }
  const auto& grant = grants_[rank];
  auto grant_root = compile_deepseek_rank_materialization_grant_root(grant);
  if (!grant_root.ok()) return grant_root.status();
  if (ack.protocol_version != 1 ||
      ack.engine_epoch != grant.engine_epoch ||
      ack.worker_generation != grant.worker_generation ||
      ack.world_size != grant.world_size || ack.rank != grant.rank ||
      ack.process_manifest_identity != grant.process_manifest_identity ||
      ack.process_identity != grant.process_identity ||
      ack.pidfd_identity != grant.pidfd_identity ||
      ack.control_identity != grant.control_identity ||
      ack.challenge_identity != grant.challenge_identity ||
      ack.grant_root != *grant_root || ack.report_root != grant.report_root ||
      ack.mapping_owner_root != grant.mapping_owner_root ||
      ack.post_mapping_seal_root != grant.post_mapping_seal_root) {
    return Status::FailedPrecondition(
        "DeepSeek materialization ACK differs from accepted grant");
  }
  return Status::Ok();
}

std::size_t
DeepSeekRankMaterializationGrantCoordinator::acknowledgment_count()
    const noexcept {
  std::size_t count = 0;
  for (const auto& acknowledgment : acknowledgments_) {
    if (acknowledgment) ++count;
  }
  return count;
}

const DeepSeekRankMaterializationGrantFields*
DeepSeekRankMaterializationGrantCoordinator::grant(
    std::uint32_t rank) const noexcept {
  return rank < grants_.size() ? &grants_[rank] : nullptr;
}

const DeepSeekRankMaterializationGrantAckFields*
DeepSeekRankMaterializationGrantCoordinator::acknowledgment(
    std::uint32_t rank) const noexcept {
  return complete_ && rank < acknowledgments_.size() &&
                 acknowledgments_[rank]
             ? &*acknowledgments_[rank]
             : nullptr;
}

Status DeepSeekRankMaterializationGrantCoordinator::advance() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek materialization grant coordinator is poisoned");
  }
  if (complete_) return Status::Ok();
  auto status = ensure_before_deadline();
  if (!status.ok()) return fail(status);
  status = supervisor_->poll();
  if (!status.ok()) return fail(status);

  bool pending = false;
  for (std::uint32_t rank = 0; rank < manifests_.size(); ++rank) {
    const auto* handle = supervisor_->process_handle(rank);
    if (handle == nullptr) {
      return fail(Status::FailedPrecondition(
          "DeepSeek materialization process handle disappeared"));
    }
    if (!grants_dispatched_[rank]) {
      status = ensure_before_deadline();
      if (!status.ok()) return fail(status);
      status = channel_->send_grant(*handle, frames_[rank]);
      if (!status.ok()) {
        if (status.code() == StatusCode::kUnavailable) {
          pending = true;
          continue;
        }
        return fail(status);
      }
      status = ensure_before_deadline();
      if (!status.ok()) return fail(status);
      grants_dispatched_[rank] = true;
    }
    if (acknowledgments_[rank]) continue;
    status = ensure_before_deadline();
    if (!status.ok()) return fail(status);
    auto frame = channel_->poll_ack(*handle);
    if (!frame.ok()) {
      if (frame.status().code() == StatusCode::kUnavailable) {
        pending = true;
        continue;
      }
      return fail(frame.status());
    }
    if (!frame->has_value()) {
      pending = true;
      continue;
    }
    status = ensure_before_deadline();
    if (!status.ok()) return fail(status);
    auto ack = decode_deepseek_rank_materialization_grant_ack(**frame);
    if (!ack.ok()) return fail(ack.status());
    status = validate_ack(rank, *ack);
    if (!status.ok()) return fail(status);
    acknowledgments_[rank] = std::move(*ack);
  }
  if (pending || acknowledgment_count() != manifests_.size()) {
    return Status::Unavailable(
        "DeepSeek materialization acknowledgments are pending");
  }
  complete_ = true;
  return Status::Ok();
}

}  // namespace pih
