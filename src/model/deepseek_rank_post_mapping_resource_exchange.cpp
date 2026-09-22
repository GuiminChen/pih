#include "pih/model/deepseek_rank_post_mapping_resource_exchange.h"

#include <utility>

namespace pih {
namespace {

bool manifest_matches_ready(
    const DeepSeekRankProcessManifest& manifest,
    const DeepSeekRankExecReady& ready) noexcept {
  const auto& receipt = ready.receipt;
  return ready.challenge_identity != 0 &&
         receipt.engine_epoch == manifest.engine_epoch &&
         receipt.worker_generation == manifest.worker_generation &&
         receipt.rank == manifest.rank &&
         receipt.physical_device_identity ==
             manifest.physical_device_identity &&
         receipt.process_manifest_identity ==
             manifest.process_manifest_identity &&
         receipt.process_identity != 0 && receipt.pidfd_identity != 0 &&
         receipt.control_identity != 0 &&
         receipt.physical_device_uuid_commitment ==
             manifest.physical_device_uuid_commitment &&
         receipt.startup_device_ordinal == manifest.startup_device_ordinal &&
         receipt.startup_deadline_ns == manifest.startup_deadline_ns;
}

Status validate_reporter_authority(
    const DeepSeekRankProcessManifest& manifest,
    const DeepSeekRankExecReady& ready,
    const DeepSeekRankArtifactMappingOwner& mapping_owner,
    const DeepSeekRankPostMappingResourceAuthority& authority) {
  const auto& receipt = ready.receipt;
  if (!manifest_matches_ready(manifest, ready) ||
      authority.protocol_version != 1 ||
      authority.engine_epoch != manifest.engine_epoch ||
      authority.worker_generation != manifest.worker_generation ||
      authority.world_size != manifest.world_size ||
      authority.rank != manifest.rank ||
      authority.process_manifest_identity !=
          manifest.process_manifest_identity ||
      authority.process_identity != receipt.process_identity ||
      authority.pidfd_identity != receipt.pidfd_identity ||
      authority.control_identity != receipt.control_identity ||
      authority.challenge_identity != ready.challenge_identity ||
      mapping_owner.engine_epoch() != manifest.engine_epoch ||
      mapping_owner.worker_generation() != manifest.worker_generation ||
      mapping_owner.world_size() != manifest.world_size ||
      mapping_owner.rank() != manifest.rank ||
      authority.descriptor_transaction_root !=
          mapping_owner.descriptor_transaction_root() ||
      authority.metadata_transaction_root !=
          mapping_owner.metadata_transaction_root() ||
      authority.metadata_root != mapping_owner.metadata_root() ||
      authority.expected_mapping_owner_root !=
          mapping_owner.mapping_owner_root() ||
      authority.expected_mapped_interval_bytes !=
          mapping_owner.mapped_interval_bytes() ||
      authority.expected_immutability_mode !=
          mapping_owner.immutability_mode() ||
      authority.expected_source_catalog_production_eligible !=
          mapping_owner.source_catalog_production_eligible() ||
      authority.dspark_enabled != mapping_owner.dspark_enabled()) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping reporter authority differs");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekRankPostMappingResourceAuthority>
compile_deepseek_rank_post_mapping_resource_authority(
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> resource_plans,
    const DeepSeekRankPostExecResourceSeal& first_resource_seal,
    const DeepSeekRankArtifactMetadataTransferTransaction&
        metadata_transaction,
    std::uint32_t rank, std::uint64_t deadline_ns) {
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto spawn_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, spawn_plan);
  if (!spawn_root.ok()) return spawn_root.status();
  auto plan_set_root =
      compile_deepseek_rank_post_exec_resource_plan_set_root(
          manifests, spawn_plan, resource_plans);
  if (!plan_set_root.ok()) return plan_set_root.status();
  if (!supervisor.ready() || supervisor.failed() ||
      rank >= manifests.size() || resource_plans.size() != manifests.size() ||
      metadata_transaction.world_size() != manifests.size() ||
      !metadata_transaction.complete() || metadata_transaction.poisoned() ||
      !metadata_transaction.retains_descriptor_transaction() ||
      deadline_ns <= metadata_transaction.deadline_ns() ||
      *manifest_root != supervisor.manifest_root() ||
      *spawn_root != supervisor.spawn_resource_plan_root() ||
      first_resource_seal.engine_epoch() != manifests[rank].engine_epoch ||
      first_resource_seal.worker_generation() !=
          manifests[rank].worker_generation ||
      first_resource_seal.world_size() != manifests.size() ||
      first_resource_seal.capacity_plan_instance_root() !=
          supervisor.capacity_plan_instance_root() ||
      first_resource_seal.os_resource_envelope_root() !=
          resource_plans[rank].os_resource_envelope_root ||
      first_resource_seal.plan_set_root() != *plan_set_root) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping resource authority cannot be compiled");
  }
  const auto* ready = supervisor.exec_ready(rank);
  if (ready == nullptr || !manifest_matches_ready(manifests[rank], *ready)) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping exec identity is absent");
  }
  const auto& manifest = manifests[rank];
  DeepSeekRankPostMappingResourceAuthority authority{
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
      resource_plans[rank].os_resource_envelope_root,
      first_resource_seal.seal_root(),
      first_resource_seal.plan_set_root(),
      metadata_transaction.descriptor_transaction_root(),
      metadata_transaction.transaction_root(),
      metadata_transaction.metadata_root(rank),
      metadata_transaction.expected_mapping_owner_root(rank),
      deadline_ns,
      metadata_transaction.expected_mapped_interval_bytes(rank),
      metadata_transaction.expected_immutability_mode(rank),
      metadata_transaction
          .expected_source_catalog_production_eligible(rank),
      metadata_transaction.dspark_enabled(),
      resource_plans[rank]};
  auto root =
      compile_deepseek_rank_post_mapping_resource_authority_frame_root(
          authority);
  if (!root.ok()) return root.status();
  return authority;
}

DeepSeekRankPostMappingResourceReporter::
    DeepSeekRankPostMappingResourceReporter(
        DeepSeekRankProcessManifest manifest,
        DeepSeekRankExecReady exec_ready, std::int32_t control_fd,
        const DeepSeekRankArtifactMappingOwner& mapping_owner,
        StableDeepSeekRankPostExecResourceCollector& collector,
        DeepSeekRankPostMappingResourceReporterOperations& operations) noexcept
    : manifest_(std::move(manifest)), exec_ready_(std::move(exec_ready)),
      control_fd_(control_fd), mapping_owner_(&mapping_owner),
      collector_(&collector), operations_(&operations) {}

Result<DeepSeekRankPostMappingResourceReporter>
DeepSeekRankPostMappingResourceReporter::Create(
    DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
    std::int32_t control_fd,
    const DeepSeekRankArtifactMappingOwner& mapping_owner,
    StableDeepSeekRankPostExecResourceCollector& collector,
    DeepSeekRankPostMappingResourceReporterOperations& operations) {
  if (!manifest_matches_ready(manifest, exec_ready) || control_fd < 0 ||
      mapping_owner.engine_epoch() != manifest.engine_epoch ||
      mapping_owner.worker_generation() != manifest.worker_generation ||
      mapping_owner.world_size() != manifest.world_size ||
      mapping_owner.rank() != manifest.rank ||
      mapping_owner.mapping_owner_root() == Sha256Digest{}) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping reporter identity is invalid");
  }
  return DeepSeekRankPostMappingResourceReporter(
      std::move(manifest), std::move(exec_ready), control_fd,
      mapping_owner, collector, operations);
}

Status DeepSeekRankPostMappingResourceReporter::fail(
    Status cause) noexcept {
  poisoned_ = true;
  observation_frame_.reset();
  report_.reset();
  return cause.ok()
             ? Status::Internal(
                   "DeepSeek post-mapping resource reporter poisoned")
             : cause;
}

Status DeepSeekRankPostMappingResourceReporter::advance() {
  if (reported_ || poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping resource reporter is closed");
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
          "DeepSeek post-mapping resource authority is pending");
    }
    auto authority = decode_deepseek_rank_post_mapping_resource_authority(
        **received);
    if (!authority.ok()) return fail(authority.status());
    auto status = validate_reporter_authority(
        manifest_, exec_ready_, *mapping_owner_, *authority);
    if (!status.ok()) return fail(status);
    authority_ = std::move(*authority);
  }
  auto now = operations_->monotonic_now_ns();
  if (!now.ok()) return fail(now.status());
  if (*now >= authority_->deadline_ns) {
    return fail(Status::DeadlineExceeded(
        "DeepSeek post-mapping resource reporter deadline expired"));
  }
  if (!observation_frame_) {
    const DeepSeekRankPostExecResourceIdentity identity{
        authority_->engine_epoch,
        authority_->worker_generation,
        authority_->rank,
        authority_->process_identity,
        authority_->challenge_identity,
        authority_->capacity_plan_instance_root,
        authority_->os_resource_envelope_root};
    auto resources = collector_->collect(identity);
    if (!resources.ok()) return fail(resources.status());
    DeepSeekRankPostMappingResourceObservation observation{
        *resources,
        authority_->first_resource_seal_root,
        mapping_owner_->descriptor_transaction_root(),
        mapping_owner_->metadata_transaction_root(),
        mapping_owner_->metadata_root(),
        mapping_owner_->mapping_owner_root(),
        mapping_owner_->mapped_interval_bytes(),
        mapping_owner_->immutability_mode(),
        mapping_owner_->source_catalog_production_eligible(),
        mapping_owner_->dspark_enabled()};
    auto report = DeepSeekRankPostMappingResourceReport::Compile(
        *authority_, observation);
    if (!report.ok()) return fail(report.status());
    auto frame = encode_deepseek_rank_post_mapping_resource_observation(
        observation);
    if (!frame.ok()) return fail(frame.status());
    report_ = std::move(*report);
    observation_frame_ = std::move(*frame);
    now = operations_->monotonic_now_ns();
    if (!now.ok()) return fail(now.status());
    if (*now >= authority_->deadline_ns) {
      return fail(Status::DeadlineExceeded(
          "DeepSeek post-mapping resource collection exceeded deadline"));
    }
  }
  auto status = operations_->send_observation(
      control_fd_, *observation_frame_);
  if (!status.ok()) {
    if (status.code() == StatusCode::kUnavailable) return status;
    return fail(status);
  }
  now = operations_->monotonic_now_ns();
  if (!now.ok()) return fail(now.status());
  if (*now >= authority_->deadline_ns) {
    return fail(Status::DeadlineExceeded(
        "DeepSeek post-mapping resource send crossed deadline"));
  }
  reported_ = true;
  observation_frame_.reset();
  return Status::Ok();
}

std::optional<DeepSeekRankPostMappingResourceReporterWaitEvent>
DeepSeekRankPostMappingResourceReporter::wait_event() const noexcept {
  if (reported_ || poisoned_) return std::nullopt;
  return authority_
             ? DeepSeekRankPostMappingResourceReporterWaitEvent::
                   kObservationWritable
             : DeepSeekRankPostMappingResourceReporterWaitEvent::
                   kAuthorityReadable;
}

std::optional<std::uint64_t>
DeepSeekRankPostMappingResourceReporter::deadline_ns() const noexcept {
  return authority_ ? std::optional<std::uint64_t>{authority_->deadline_ns}
                    : std::nullopt;
}

DeepSeekRankPostMappingResourceCoordinator::
    DeepSeekRankPostMappingResourceCoordinator(
        DeepSeekRankProcessSupervisor& supervisor,
        std::vector<DeepSeekRankProcessManifest> manifests,
        DeepSeekRankSpawnResourcePlan spawn_plan,
        std::vector<DeepSeekRankPostExecResourcePlan> resource_plans,
        DeepSeekRankPostExecResourceSeal first_resource_seal,
        const DeepSeekRankArtifactMetadataTransferTransaction&
            metadata_transaction,
        std::vector<DeepSeekRankPostMappingResourceAuthority> authorities,
        std::vector<std::array<
            std::byte,
            kDeepSeekRankPostMappingResourceAuthorityFrameBytes>>
            authority_frames,
        std::uint64_t deadline_ns,
        DeepSeekRankPostMappingResourceChannel& channel) noexcept
    : supervisor_(&supervisor), manifests_(std::move(manifests)),
      spawn_plan_(spawn_plan), resource_plans_(std::move(resource_plans)),
      first_resource_seal_(std::move(first_resource_seal)),
      metadata_transaction_(&metadata_transaction),
      authorities_(std::move(authorities)),
      authority_frames_(std::move(authority_frames)),
      authorities_dispatched_(manifests_.size(), false),
      reports_(manifests_.size()), receipts_(manifests_.size()),
      deadline_ns_(deadline_ns),
      channel_(&channel) {}

Result<DeepSeekRankPostMappingResourceCoordinator>
DeepSeekRankPostMappingResourceCoordinator::Create(
    DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnResourcePlan spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> resource_plans,
    DeepSeekRankPostExecResourceSeal first_resource_seal,
    const DeepSeekRankArtifactMetadataTransferTransaction&
        metadata_transaction,
    std::uint64_t deadline_ns,
    DeepSeekRankPostMappingResourceChannel& channel) {
  const auto fail_create =
      [&supervisor](Status cause)
      -> Result<DeepSeekRankPostMappingResourceCoordinator> {
    if (cause.ok()) {
      cause = Status::Internal(
          "DeepSeek post-mapping coordinator construction failed");
    }
    if (supervisor.ready() && !supervisor.failed()) {
      (void)supervisor.abort_post_exec(cause);
    }
    return cause;
  };
  if (manifests.empty() || manifests.size() > 4 ||
      resource_plans.size() != manifests.size() ||
      deadline_ns <= metadata_transaction.deadline_ns()) {
    return fail_create(Status::InvalidArgument(
        "DeepSeek post-mapping coordinator inputs are invalid"));
  }
  std::vector<std::array<
      std::byte, kDeepSeekRankPostMappingResourceAuthorityFrameBytes>>
      frames;
  std::vector<DeepSeekRankPostMappingResourceAuthority> authorities;
  frames.reserve(manifests.size());
  authorities.reserve(manifests.size());
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    auto authority = compile_deepseek_rank_post_mapping_resource_authority(
        supervisor, manifests, spawn_plan, resource_plans,
        first_resource_seal, metadata_transaction, rank, deadline_ns);
    if (!authority.ok()) return fail_create(authority.status());
    auto frame = encode_deepseek_rank_post_mapping_resource_authority(
        *authority);
    if (!frame.ok()) return fail_create(frame.status());
    authorities.push_back(std::move(*authority));
    frames.push_back(*frame);
  }
  return DeepSeekRankPostMappingResourceCoordinator(
      supervisor,
      std::vector<DeepSeekRankProcessManifest>(manifests.begin(),
                                               manifests.end()),
      spawn_plan,
      std::vector<DeepSeekRankPostExecResourcePlan>(
          resource_plans.begin(), resource_plans.end()),
      std::move(first_resource_seal), metadata_transaction,
      std::move(authorities), std::move(frames), deadline_ns, channel);
}

Status DeepSeekRankPostMappingResourceCoordinator::fail(
    Status cause) noexcept {
  poisoned_ = true;
  if (!supervisor_->failed()) {
    return supervisor_->abort_post_exec(
        cause.ok()
            ? Status::Internal(
                  "DeepSeek post-mapping resource coordination failed")
            : cause);
  }
  return cause.ok()
             ? Status::Internal(
                   "DeepSeek post-mapping resource coordination failed")
             : cause;
}

Status DeepSeekRankPostMappingResourceCoordinator::
    ensure_before_deadline() {
  auto now = channel_->monotonic_now_ns();
  if (!now.ok()) return now.status();
  if (*now >= deadline_ns_) {
    return Status::DeadlineExceeded(
        "DeepSeek post-mapping resource deadline expired");
  }
  return Status::Ok();
}

std::size_t DeepSeekRankPostMappingResourceCoordinator::receipt_count()
    const noexcept {
  std::size_t count = 0;
  for (const auto& receipt : receipts_) {
    if (receipt) ++count;
  }
  return count;
}

Status DeepSeekRankPostMappingResourceCoordinator::advance() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping resource coordinator is poisoned");
  }
  if (seal_) return Status::Ok();
  auto status = ensure_before_deadline();
  if (!status.ok()) return fail(status);
  status = supervisor_->poll();
  if (!status.ok()) return fail(status);
  bool pending = false;
  for (std::uint32_t rank = 0; rank < manifests_.size(); ++rank) {
    const auto* handle = supervisor_->process_handle(rank);
    if (handle == nullptr) {
      return fail(Status::FailedPrecondition(
          "DeepSeek post-mapping process handle disappeared"));
    }
    if (!authorities_dispatched_[rank]) {
      status = ensure_before_deadline();
      if (!status.ok()) return fail(status);
      status = channel_->send_authority(
          *handle, authority_frames_[rank]);
      if (!status.ok()) {
        if (status.code() == StatusCode::kUnavailable) {
          pending = true;
          continue;
        }
        return fail(status);
      }
      status = ensure_before_deadline();
      if (!status.ok()) return fail(status);
      authorities_dispatched_[rank] = true;
    }
    if (receipts_[rank]) continue;
    status = ensure_before_deadline();
    if (!status.ok()) return fail(status);
    auto frame = channel_->poll_observation(*handle);
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
    auto observation =
        decode_deepseek_rank_post_mapping_resource_observation(**frame);
    if (!observation.ok()) return fail(observation.status());
    auto report = DeepSeekRankPostMappingResourceReport::Compile(
        authorities_[rank], *observation);
    if (!report.ok()) return fail(report.status());
    auto receipt = DeepSeekRankPostMappingResourceReceipt::Compile(
        *supervisor_, manifests_, spawn_plan_, resource_plans_,
        first_resource_seal_, *metadata_transaction_, *observation);
    if (!receipt.ok()) return fail(receipt.status());
    reports_[rank] = std::move(*report);
    receipts_[rank] = std::move(*receipt);
  }
  if (pending || receipt_count() != receipts_.size()) {
    return Status::Unavailable(
        "DeepSeek post-mapping resource observations are pending");
  }
  std::vector<DeepSeekRankPostMappingResourceReceipt> receipts;
  receipts.reserve(receipts_.size());
  for (const auto& receipt : receipts_) {
    if (!receipt) {
      return fail(Status::FailedPrecondition(
          "DeepSeek post-mapping receipt set is incomplete"));
    }
    receipts.push_back(*receipt);
  }
  auto seal = DeepSeekRankPostMappingResourceSeal::Compile(
      *supervisor_, manifests_, spawn_plan_, resource_plans_,
      first_resource_seal_, *metadata_transaction_, receipts);
  if (!seal.ok()) return fail(seal.status());
  seal_ = std::move(*seal);
  return Status::Ok();
}

}  // namespace pih
