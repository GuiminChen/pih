#include "pih/model/deepseek_rank_post_exec_resource_exchange.h"

#include <utility>

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) noexcept {
  for (const auto byte : value.bytes) {
    if (byte != std::byte{0}) return true;
  }
  return false;
}

bool manifest_matches_ready(
    const DeepSeekRankProcessManifest& manifest,
    const DeepSeekRankExecReady& ready) noexcept {
  const auto& receipt = ready.receipt;
  return manifest.engine_epoch != 0 && manifest.worker_generation != 0 &&
         manifest.world_size > 0 && manifest.world_size <= 4 &&
         manifest.rank < manifest.world_size &&
         manifest.engine_epoch == receipt.engine_epoch &&
         manifest.worker_generation == receipt.worker_generation &&
         manifest.rank == receipt.rank &&
         manifest.physical_device_identity ==
             receipt.physical_device_identity &&
         manifest.process_manifest_identity ==
             receipt.process_manifest_identity &&
         manifest.startup_device_ordinal ==
             receipt.startup_device_ordinal &&
         manifest.startup_deadline_ns == receipt.startup_deadline_ns &&
         receipt.process_identity != 0 && receipt.pidfd_identity != 0 &&
         receipt.control_identity != 0 && ready.challenge_identity != 0 &&
         receipt.physical_device_uuid_commitment != Sha256Digest{};
}

Status validate_reporter_authority(
    const DeepSeekRankProcessManifest& manifest,
    const DeepSeekRankExecReady& ready,
    const DeepSeekRankPostExecResourceAuthority& authority) {
  const auto& receipt = ready.receipt;
  const auto& plan = authority.resource_plan;
  if (authority.protocol_version != 1 ||
      authority.engine_epoch != receipt.engine_epoch ||
      authority.worker_generation != receipt.worker_generation ||
      authority.world_size != manifest.world_size ||
      authority.rank != receipt.rank ||
      authority.process_manifest_identity !=
          receipt.process_manifest_identity ||
      authority.process_identity != receipt.process_identity ||
      authority.pidfd_identity != receipt.pidfd_identity ||
      authority.control_identity != receipt.control_identity ||
      authority.challenge_identity != ready.challenge_identity ||
      !nonzero(authority.manifest_root) ||
      !nonzero(authority.spawn_resource_plan_root) ||
      !nonzero(authority.capacity_plan_instance_root) ||
      !nonzero(authority.os_resource_envelope_root) ||
      authority.deadline_ns <= manifest.startup_deadline_ns ||
      plan.worker_task_peak == 0 || plan.worker_fd_peak == 0 ||
      plan.worker_vma_peak == 0 || plan.task_emergency_reserve == 0 ||
      plan.fd_emergency_reserve == 0 ||
      plan.vma_emergency_reserve == 0 ||
      plan.os_resource_envelope_root !=
          authority.os_resource_envelope_root) {
    return Status::FailedPrecondition(
        "DeepSeek rank post-exec resource authority is invalid");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekRankPostExecResourceAuthority>
compile_deepseek_rank_post_exec_resource_authority(
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> plans,
    std::uint32_t rank, std::uint64_t deadline_ns) {
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto spawn_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, spawn_plan);
  if (!spawn_root.ok()) return spawn_root.status();
  if (!supervisor.ready() || supervisor.failed() ||
      rank >= manifests.size() || plans.size() != manifests.size() ||
      deadline_ns <= manifests[rank].startup_deadline_ns ||
      *manifest_root != supervisor.manifest_root() ||
      *spawn_root != supervisor.spawn_resource_plan_root() ||
      !nonzero(supervisor.capacity_plan_instance_root())) {
    return Status::FailedPrecondition(
        "DeepSeek rank post-exec authority cannot be compiled");
  }
  auto plan_root = compile_deepseek_rank_post_exec_resource_plan_root(
      manifests, spawn_plan, rank, plans[rank]);
  if (!plan_root.ok()) return plan_root.status();
  const auto* ready = supervisor.exec_ready(rank);
  if (ready == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank exec-ready identity is absent");
  }
  const auto& manifest = manifests[rank];
  return DeepSeekRankPostExecResourceAuthority{
      1,
      manifest.engine_epoch,
      manifest.worker_generation,
      manifest.world_size,
      manifest.rank,
      manifest.process_manifest_identity,
      ready->receipt.process_identity,
      ready->receipt.pidfd_identity,
      ready->receipt.control_identity,
      ready->challenge_identity,
      *manifest_root,
      *spawn_root,
      supervisor.capacity_plan_instance_root(),
      plans[rank].os_resource_envelope_root,
      deadline_ns,
      plans[rank]};
}

DeepSeekRankPostExecResourceReporter::DeepSeekRankPostExecResourceReporter(
    DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
    std::int32_t control_fd,
    StableDeepSeekRankPostExecResourceCollector& collector,
    DeepSeekRankPostExecResourceReporterOperations& operations) noexcept
    : manifest_(std::move(manifest)), exec_ready_(std::move(exec_ready)),
      control_fd_(control_fd), collector_(&collector),
      operations_(&operations) {}

Result<DeepSeekRankPostExecResourceReporter>
DeepSeekRankPostExecResourceReporter::Create(
    DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
    std::int32_t control_fd,
    StableDeepSeekRankPostExecResourceCollector& collector,
    DeepSeekRankPostExecResourceReporterOperations& operations) {
  if (!manifest_matches_ready(manifest, exec_ready) || control_fd < 0) {
    return Status::InvalidArgument(
        "DeepSeek rank post-exec reporter identity is invalid");
  }
  return DeepSeekRankPostExecResourceReporter(
      std::move(manifest), std::move(exec_ready), control_fd, collector,
      operations);
}

Status DeepSeekRankPostExecResourceReporter::fail(Status cause) noexcept {
  poisoned_ = true;
  frame_.reset();
  return cause.ok()
             ? Status::Internal(
                   "DeepSeek rank post-exec resource reporter poisoned")
             : cause;
}

Status DeepSeekRankPostExecResourceReporter::advance() {
  if (reported_ || poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek rank post-exec resource reporter is closed");
  }
  if (!authority_) {
    auto received = operations_->receive_authority(control_fd_);
    if (!received.ok()) {
      if (received.status().code() == StatusCode::kUnavailable) {
        return received.status();
      }
      return fail(received.status());
    }
    if (!received->has_value()) {
      return Status::Unavailable(
          "DeepSeek rank post-exec resource authority is pending");
    }
    auto authority = decode_deepseek_rank_post_exec_resource_authority(
        **received);
    if (!authority.ok()) return fail(authority.status());
    auto status = validate_reporter_authority(
        manifest_, exec_ready_, *authority);
    if (!status.ok()) return fail(status);
    authority_ = std::move(*authority);
  }
  auto now = operations_->monotonic_now_ns();
  if (!now.ok()) return fail(now.status());
  if (*now >= authority_->deadline_ns) {
    return fail(Status::DeadlineExceeded(
        "DeepSeek rank post-exec resource reporter deadline expired"));
  }
  if (!frame_) {
    const DeepSeekRankPostExecResourceIdentity identity{
        authority_->engine_epoch,
        authority_->worker_generation,
        authority_->rank,
        authority_->process_identity,
        authority_->challenge_identity,
        authority_->capacity_plan_instance_root,
        authority_->os_resource_envelope_root};
    auto observation = collector_->collect(identity);
    if (!observation.ok()) return fail(observation.status());
    frame_ = encode_deepseek_rank_post_exec_resource_observation(
        *observation);
    now = operations_->monotonic_now_ns();
    if (!now.ok()) return fail(now.status());
    if (*now >= authority_->deadline_ns) {
      return fail(Status::DeadlineExceeded(
          "DeepSeek rank post-exec resource collection exceeded deadline"));
    }
  }
  const auto status = operations_->send_observation(control_fd_, *frame_);
  if (!status.ok()) {
    if (status.code() == StatusCode::kUnavailable) return status;
    return fail(status);
  }
  reported_ = true;
  frame_.reset();
  return Status::Ok();
}

std::optional<DeepSeekRankPostExecResourceReporterWaitEvent>
DeepSeekRankPostExecResourceReporter::wait_event() const noexcept {
  if (reported_ || poisoned_) return std::nullopt;
  return authority_
             ? DeepSeekRankPostExecResourceReporterWaitEvent::
                   kObservationWritable
             : DeepSeekRankPostExecResourceReporterWaitEvent::
                   kAuthorityReadable;
}

std::optional<std::uint64_t>
DeepSeekRankPostExecResourceReporter::deadline_ns() const noexcept {
  return authority_ ? std::optional<std::uint64_t>{authority_->deadline_ns}
                    : std::nullopt;
}

DeepSeekRankPostExecResourceCoordinator::
    DeepSeekRankPostExecResourceCoordinator(
        DeepSeekRankProcessSupervisor& supervisor,
        std::vector<DeepSeekRankProcessManifest> manifests,
        DeepSeekRankSpawnResourcePlan spawn_plan,
        std::vector<DeepSeekRankPostExecResourcePlan> plans,
        std::vector<std::array<
            std::byte, kDeepSeekRankPostExecResourceAuthorityBytes>>
            authority_frames,
        std::uint64_t deadline_ns,
        DeepSeekRankPostExecResourceChannel& channel) noexcept
    : supervisor_(&supervisor), manifests_(std::move(manifests)),
      spawn_plan_(spawn_plan), plans_(std::move(plans)),
      authority_frames_(std::move(authority_frames)),
      deadline_ns_(deadline_ns), channel_(&channel),
      receipts_(manifests_.size()) {}

Result<DeepSeekRankPostExecResourceCoordinator>
DeepSeekRankPostExecResourceCoordinator::Create(
    DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnResourcePlan spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> plans,
    std::uint64_t deadline_ns,
    DeepSeekRankPostExecResourceChannel& channel) {
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto spawn_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, spawn_plan);
  if (!spawn_root.ok()) return spawn_root.status();
  if (!supervisor.ready() || supervisor.failed() ||
      plans.size() != manifests.size() || deadline_ns == 0 ||
      deadline_ns <= manifests[0].startup_deadline_ns ||
      *manifest_root != supervisor.manifest_root() ||
      *spawn_root != supervisor.spawn_resource_plan_root()) {
    return Status::FailedPrecondition(
        "DeepSeek rank post-exec coordinator authority is invalid");
  }
  std::vector<std::array<
      std::byte, kDeepSeekRankPostExecResourceAuthorityBytes>>
      authority_frames;
  authority_frames.reserve(manifests.size());
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    if (supervisor.process_handle(rank) == nullptr ||
        supervisor.exec_ready(rank) == nullptr ||
        plans[rank].os_resource_envelope_root !=
            plans[0].os_resource_envelope_root) {
      return Status::FailedPrecondition(
          "DeepSeek rank post-exec coordinator plan is inconsistent");
    }
    auto plan_root = compile_deepseek_rank_post_exec_resource_plan_root(
        manifests, spawn_plan, rank, plans[rank]);
    if (!plan_root.ok()) return plan_root.status();
    auto authority = compile_deepseek_rank_post_exec_resource_authority(
        supervisor, manifests, spawn_plan, plans, rank, deadline_ns);
    if (!authority.ok()) return authority.status();
    authority_frames.push_back(
        encode_deepseek_rank_post_exec_resource_authority(*authority));
  }
  return DeepSeekRankPostExecResourceCoordinator(
      supervisor,
      std::vector<DeepSeekRankProcessManifest>(manifests.begin(),
                                               manifests.end()),
      spawn_plan,
      std::vector<DeepSeekRankPostExecResourcePlan>(plans.begin(),
                                                   plans.end()),
      std::move(authority_frames), deadline_ns, channel);
}

Status DeepSeekRankPostExecResourceCoordinator::fail(
    Status cause) noexcept {
  poisoned_ = true;
  if (!supervisor_->failed()) {
    return supervisor_->abort_post_exec(
        cause.ok()
            ? Status::Internal(
                  "DeepSeek rank post-exec resource coordination failed")
            : cause);
  }
  return cause.ok()
             ? Status::Internal(
                   "DeepSeek rank post-exec resource coordination failed")
             : cause;
}

std::size_t DeepSeekRankPostExecResourceCoordinator::receipt_count()
    const noexcept {
  std::size_t count = 0;
  for (const auto& receipt : receipts_) {
    if (receipt) ++count;
  }
  return count;
}

Status DeepSeekRankPostExecResourceCoordinator::advance(
    std::uint64_t now_ns) {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek rank post-exec resource coordinator is poisoned");
  }
  if (seal_) return Status::Ok();
  if (now_ns >= deadline_ns_) {
    return fail(Status::DeadlineExceeded(
        "DeepSeek rank post-exec resource deadline expired"));
  }
  auto status = supervisor_->poll();
  if (!status.ok()) return fail(status);
  if (!authorities_dispatched_) {
    for (std::uint32_t rank = 0; rank < manifests_.size(); ++rank) {
      const auto* handle = supervisor_->process_handle(rank);
      if (handle == nullptr) {
        return fail(Status::FailedPrecondition(
            "DeepSeek rank post-exec authority handle disappeared"));
      }
      status = channel_->send_authority(
          *handle, authority_frames_[rank]);
      if (!status.ok()) return fail(status);
    }
    authorities_dispatched_ = true;
  }
  bool pending = false;
  for (std::uint32_t rank = 0; rank < manifests_.size(); ++rank) {
    if (receipts_[rank]) continue;
    const auto* handle = supervisor_->process_handle(rank);
    const auto* ready = supervisor_->exec_ready(rank);
    if (handle == nullptr || ready == nullptr) {
      return fail(Status::FailedPrecondition(
          "DeepSeek rank post-exec process identity disappeared"));
    }
    auto observation = channel_->poll_observation(*handle);
    if (!observation.ok()) return fail(observation.status());
    if (!observation->has_value()) {
      pending = true;
      continue;
    }
    auto receipt = DeepSeekRankPostExecResourceReceipt::Compile(
        manifests_, spawn_plan_, plans_[rank], *ready, **observation);
    if (!receipt.ok()) return fail(receipt.status());
    receipts_[rank] = std::move(*receipt);
  }
  if (pending) {
    return Status::Unavailable(
        "DeepSeek rank post-exec resource observations are pending");
  }
  std::vector<DeepSeekRankPostExecResourceReceipt> receipts;
  receipts.reserve(receipts_.size());
  for (const auto& receipt : receipts_) {
    if (!receipt) {
      return fail(Status::FailedPrecondition(
          "DeepSeek rank post-exec receipt set is incomplete"));
    }
    receipts.push_back(*receipt);
  }
  auto seal = DeepSeekRankPostExecResourceSeal::Compile(
      *supervisor_, manifests_, spawn_plan_, plans_, receipts);
  if (!seal.ok()) return fail(seal.status());
  seal_ = std::move(*seal);
  return Status::Ok();
}

}  // namespace pih
