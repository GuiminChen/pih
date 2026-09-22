#include "pih/model/deepseek_rank_process_supervisor.h"

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"
#include "pih/model/deepseek_rank_capacity_plan_instance.h"

#include <set>
#include <utility>

namespace pih {
namespace {

Status validate_manifests(
    std::span<const DeepSeekRankProcessManifest> manifests) {
  if (manifests.empty() || manifests.size() > 4) {
    return Status::InvalidArgument("DeepSeek rank process count is invalid");
  }
  std::set<std::uint64_t> devices;
  std::set<std::array<std::byte, 32>> device_commitments;
  std::set<std::int32_t> startup_ordinals;
  const auto epoch = manifests[0].engine_epoch;
  const auto generation = manifests[0].worker_generation;
  for (std::size_t index = 0; index < manifests.size(); ++index) {
    const auto& manifest = manifests[index];
    if (manifest.engine_epoch == 0 || manifest.worker_generation == 0 ||
        manifest.engine_epoch != epoch ||
        manifest.worker_generation != generation ||
        manifest.world_size != manifests.size() || manifest.rank != index ||
        manifest.physical_device_identity == 0 ||
        manifest.process_manifest_identity == 0 ||
        manifest.physical_device_uuid_commitment == Sha256Digest{} ||
        manifest.startup_device_ordinal < 0 ||
        manifest.startup_deadline_ns == 0 ||
        manifest.startup_deadline_ns != manifests[0].startup_deadline_ns ||
        !devices.insert(manifest.physical_device_identity).second ||
        !device_commitments
             .insert(manifest.physical_device_uuid_commitment.bytes)
             .second ||
        !startup_ordinals.insert(manifest.startup_device_ordinal).second) {
      return Status::InvalidArgument(
          "DeepSeek rank process manifest is invalid");
    }
  }
  return Status::Ok();
}

Result<Sha256Digest> rank_manifest_root_impl(
    std::span<const DeepSeekRankProcessManifest> manifests) {
  auto collection = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-process-manifest-set:v1",
      static_cast<std::uint32_t>(manifests.size() + 1));
  if (!collection.ok()) return collection.status();
  auto status = collection->add_u32(
      1, static_cast<std::uint32_t>(manifests.size()));
  for (std::size_t index = 0; status.ok() && index < manifests.size(); ++index) {
    const auto& manifest = manifests[index];
    auto record = CanonicalHashBuilder::Create(
        "pih:deepseek-rank-process-manifest:v1", 9);
    if (!record.ok()) return record.status();
    status = record->add_u64(1, manifest.engine_epoch);
    if (status.ok())
      status = record->add_u64(2, manifest.worker_generation);
    if (status.ok()) status = record->add_u32(3, manifest.world_size);
    if (status.ok()) status = record->add_u32(4, manifest.rank);
    if (status.ok())
      status = record->add_u64(5, manifest.physical_device_identity);
    if (status.ok())
      status = record->add_u64(6, manifest.process_manifest_identity);
    if (status.ok())
      status = record->add_hash(7, manifest.physical_device_uuid_commitment);
    if (status.ok())
      status = record->add_u32(
          8, static_cast<std::uint32_t>(manifest.startup_device_ordinal));
    if (status.ok())
      status = record->add_u64(9, manifest.startup_deadline_ns);
    if (!status.ok()) return status;
    auto root = record->finalize();
    if (!root.ok()) return root.status();
    status = collection->add_hash(
        static_cast<std::uint16_t>(index + 2), *root);
  }
  if (!status.ok()) return status;
  return collection->finalize();
}

Result<Sha256Digest> cgroup_observation_root(
    std::span<const DeepSeekRankSpawnCgroupPidsObservation> ancestors) {
  if (ancestors.empty() || ancestors.size() > 16) {
    return Status::InvalidArgument(
        "DeepSeek rank spawn cgroup ancestry is invalid");
  }
  std::set<std::array<std::byte, 32>> scopes;
  auto collection = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-spawn-cgroup-set:v1",
      static_cast<std::uint32_t>(ancestors.size() + 1));
  if (!collection.ok()) return collection.status();
  auto status = collection->add_u32(
      1, static_cast<std::uint32_t>(ancestors.size()));
  std::uint64_t child_current = 0;
  for (std::size_t index = 0;
       status.ok() && index < ancestors.size(); ++index) {
    const auto& ancestor = ancestors[index];
    if (!scopes.insert(ancestor.scope_root.bytes).second ||
        (index != 0 && ancestor.current_tasks < child_current) ||
        (ancestor.maximum_tasks &&
         ancestor.current_tasks > *ancestor.maximum_tasks)) {
      return Status::InvalidArgument(
          "DeepSeek rank spawn cgroup observation is invalid");
    }
    auto record = CanonicalHashBuilder::Create(
        "pih:deepseek-rank-spawn-cgroup:v1", 4);
    if (!record.ok()) return record.status();
    status = record->add_hash(1, ancestor.scope_root);
    if (status.ok()) status = record->add_u64(2, ancestor.current_tasks);
    if (status.ok())
      status = record->add_u32(3, ancestor.maximum_tasks ? 1U : 0U);
    if (status.ok())
      status = record->add_u64(
          4, ancestor.maximum_tasks.value_or(0));
    if (!status.ok()) return status;
    auto root = record->finalize();
    if (!root.ok()) return root.status();
    status = collection->add_hash(
        static_cast<std::uint16_t>(index + 2), *root);
    child_current = ancestor.current_tasks;
  }
  if (!status.ok()) return status;
  return collection->finalize();
}

}  // namespace

Result<Sha256Digest> compile_deepseek_rank_process_manifest_root(
    std::span<const DeepSeekRankProcessManifest> manifests) {
  return rank_manifest_root_impl(manifests);
}

Result<Sha256Digest> compile_deepseek_rank_spawn_resource_plan_root(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& plan) {
  auto status = DeepSeekRankSpawnPreflightReceipt::ValidatePlan(
      manifests, plan);
  if (!status.ok()) return status;
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-spawn-resource-plan:v1", 7);
  if (!builder.ok()) return builder.status();
  status = builder->add_u32(1, plan.worker_processes);
  if (status.ok()) status = builder->add_u32(2, plan.emergency_task_reserve);
  if (status.ok()) status = builder->add_u64(3, plan.worker_fd_peak);
  if (status.ok()) status = builder->add_u64(4, plan.fd_emergency_reserve);
  if (status.ok()) status = builder->add_u64(5, plan.worker_vma_peak);
  if (status.ok()) status = builder->add_u64(6, plan.vma_emergency_reserve);
  if (status.ok())
    status = builder->add_u64(7, plan.node_file_handle_reserve);
  if (!status.ok()) return status;
  return builder->finalize();
}

DeepSeekRankSpawnPreflightReceipt::DeepSeekRankSpawnPreflightReceipt(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t world_size, Sha256Digest manifest_root,
    std::uint64_t task_increment, std::uint64_t controller_fd_increment,
    std::uint64_t node_file_handle_increment,
    Sha256Digest resource_plan_root,
    Sha256Digest receipt_root) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      world_size_(world_size), manifest_root_(manifest_root),
      task_increment_(task_increment),
      controller_fd_increment_(controller_fd_increment),
      node_file_handle_increment_(node_file_handle_increment),
      resource_plan_root_(resource_plan_root),
      receipt_root_(receipt_root) {}

Result<DeepSeekRankSpawnPreflightReceipt>
DeepSeekRankSpawnPreflightReceipt::Compile(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& plan,
    const DeepSeekRankSpawnResourceObservation& observation) {
  auto status = ValidatePlan(manifests, plan);
  if (!status.ok()) return status;
  if (observation.rlimit_nofile_soft == 0 ||
      observation.rlimit_nofile_hard < observation.rlimit_nofile_soft ||
      observation.rlimit_nofile_hard > observation.fs_nr_open ||
      observation.uid_tasks_current == 0 ||
      observation.cgroup_ancestors.empty() ||
      observation.cgroup_ancestors.front().current_tasks == 0 ||
      observation.controller_open_fds > observation.rlimit_nofile_soft ||
      observation.node_file_allocated > observation.node_file_maximum ||
      observation.vm_max_map_count == 0 ||
      (observation.rlimit_nproc_enforced &&
       observation.rlimit_nproc_soft &&
       observation.uid_tasks_current > *observation.rlimit_nproc_soft)) {
    return Status::InvalidArgument(
        "DeepSeek rank spawn resource plan or observation is invalid");
  }
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto resource_plan_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, plan);
  if (!resource_plan_root.ok()) return resource_plan_root.status();
  auto cgroup_root = cgroup_observation_root(observation.cgroup_ancestors);
  if (!cgroup_root.ok()) return cgroup_root.status();

  auto task_increment = checked_add_u64(
      plan.worker_processes, plan.emergency_task_reserve);
  auto retained_controller_fds = checked_mul_u64(plan.worker_processes, 2);
  auto controller_spawn_fds = retained_controller_fds.ok()
      ? checked_add_u64(*retained_controller_fds, 2)
      : retained_controller_fds;
  auto controller_fd_increment = controller_spawn_fds.ok()
      ? checked_add_u64(*controller_spawn_fds, plan.fd_emergency_reserve)
      : controller_spawn_fds;
  auto retained_node_handles = checked_mul_u64(plan.worker_processes, 3);
  auto node_spawn_handles = retained_node_handles.ok()
      ? checked_add_u64(*retained_node_handles, 2)
      : retained_node_handles;
  auto node_file_handle_increment = node_spawn_handles.ok()
      ? checked_add_u64(*node_spawn_handles, plan.node_file_handle_reserve)
      : node_spawn_handles;
  auto controller_fd_peak = controller_fd_increment.ok()
      ? checked_add_u64(observation.controller_open_fds,
                        *controller_fd_increment)
      : controller_fd_increment;
  auto worker_fd_required = checked_add_u64(
      plan.worker_fd_peak, plan.fd_emergency_reserve);
  auto worker_vma_required = checked_add_u64(
      plan.worker_vma_peak, plan.vma_emergency_reserve);
  auto node_file_peak = node_file_handle_increment.ok()
      ? checked_add_u64(observation.node_file_allocated,
                        *node_file_handle_increment)
      : node_file_handle_increment;
  if (!task_increment.ok() || !controller_fd_increment.ok() ||
      !node_file_handle_increment.ok() || !controller_fd_peak.ok() ||
      !worker_fd_required.ok() || !worker_vma_required.ok() ||
      !node_file_peak.ok()) {
    return Status::ResourceExhausted(
        "DeepSeek rank spawn resource arithmetic overflowed");
  }
  if ((observation.rlimit_nproc_enforced &&
       observation.rlimit_nproc_soft &&
       *task_increment >
            *observation.rlimit_nproc_soft - observation.uid_tasks_current) ||
      *controller_fd_peak > observation.rlimit_nofile_soft ||
      *worker_fd_required > observation.rlimit_nofile_soft ||
      *worker_vma_required > observation.vm_max_map_count ||
      *node_file_peak > observation.node_file_maximum) {
    return Status::ResourceExhausted(
        "DeepSeek rank spawn resource headroom is insufficient");
  }
  for (const auto& ancestor : observation.cgroup_ancestors) {
    if (ancestor.maximum_tasks &&
        *task_increment > *ancestor.maximum_tasks - ancestor.current_tasks) {
      return Status::ResourceExhausted(
          "DeepSeek rank spawn cgroup task headroom is insufficient");
    }
  }

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-spawn-preflight:v1", 26);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, manifests[0].engine_epoch);
  if (status.ok())
    status = builder->add_u64(2, manifests[0].worker_generation);
  if (status.ok()) status = builder->add_u32(3, plan.worker_processes);
  if (status.ok()) status = builder->add_hash(4, *manifest_root);
  if (status.ok()) status = builder->add_u32(5, plan.worker_processes);
  if (status.ok()) status = builder->add_u32(6, plan.emergency_task_reserve);
  if (status.ok()) status = builder->add_u64(7, plan.worker_fd_peak);
  if (status.ok()) status = builder->add_u64(8, plan.fd_emergency_reserve);
  if (status.ok()) status = builder->add_u64(9, plan.worker_vma_peak);
  if (status.ok()) status = builder->add_u64(10, plan.vma_emergency_reserve);
  if (status.ok())
    status = builder->add_u64(11, plan.node_file_handle_reserve);
  if (status.ok())
    status = builder->add_u64(12, observation.uid_tasks_current);
  if (status.ok())
    status = builder->add_u32(
        13, observation.rlimit_nproc_soft ? 1U : 0U);
  if (status.ok())
    status = builder->add_u64(
        14, observation.rlimit_nproc_soft.value_or(0));
  if (status.ok())
    status = builder->add_u32(
        15, observation.rlimit_nproc_enforced ? 1U : 0U);
  if (status.ok()) status = builder->add_hash(16, *cgroup_root);
  if (status.ok())
    status = builder->add_u64(17, observation.controller_open_fds);
  if (status.ok())
    status = builder->add_u64(18, observation.rlimit_nofile_soft);
  if (status.ok())
    status = builder->add_u64(19, observation.rlimit_nofile_hard);
  if (status.ok()) status = builder->add_u64(20, observation.fs_nr_open);
  if (status.ok())
    status = builder->add_u64(21, observation.node_file_allocated);
  if (status.ok())
    status = builder->add_u64(22, observation.node_file_maximum);
  if (status.ok())
    status = builder->add_u64(23, observation.vm_max_map_count);
  if (status.ok()) status = builder->add_u64(24, *task_increment);
  if (status.ok()) status = builder->add_u64(25, *controller_fd_increment);
  if (status.ok())
    status = builder->add_u64(26, *node_file_handle_increment);
  if (!status.ok()) return status;
  auto receipt_root = builder->finalize();
  if (!receipt_root.ok()) return receipt_root.status();
  return DeepSeekRankSpawnPreflightReceipt(
      manifests[0].engine_epoch, manifests[0].worker_generation,
      plan.worker_processes, *manifest_root, *task_increment,
      *controller_fd_increment, *node_file_handle_increment,
      *resource_plan_root, *receipt_root);
}

Status DeepSeekRankSpawnPreflightReceipt::ValidatePlan(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& plan) {
  auto status = validate_manifests(manifests);
  if (!status.ok()) return status;
  if (plan.worker_processes != manifests.size() ||
      plan.emergency_task_reserve == 0 || plan.worker_fd_peak == 0 ||
      plan.fd_emergency_reserve == 0 || plan.worker_vma_peak == 0 ||
      plan.vma_emergency_reserve == 0 ||
      plan.node_file_handle_reserve == 0) {
    return Status::InvalidArgument(
        "DeepSeek rank spawn resource plan is invalid");
  }
  return Status::Ok();
}

DeepSeekRankSpawnAuthorization::DeepSeekRankSpawnAuthorization(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t world_size, Sha256Digest manifest_root,
    Sha256Digest capacity_plan_instance_root,
    Sha256Digest spawn_resource_plan_root,
    Sha256Digest prospective_resource_receipt_root,
    Sha256Digest authorization_root,
    std::shared_ptr<const RuntimeEngineAdmission>
        capacity_authority_anchor) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      world_size_(world_size), manifest_root_(manifest_root),
      capacity_plan_instance_root_(capacity_plan_instance_root),
      spawn_resource_plan_root_(spawn_resource_plan_root),
      prospective_resource_receipt_root_(prospective_resource_receipt_root),
      authorization_root_(authorization_root),
      capacity_authority_anchor_(std::move(capacity_authority_anchor)) {}

Result<DeepSeekRankSpawnAuthorization>
DeepSeekRankSpawnAuthorization::Create(
    std::span<const DeepSeekRankProcessManifest> manifests,
    DeepSeekRankCapacityPlanInstance capacity_plan_instance,
    const DeepSeekRankSpawnPreflightReceipt& preflight) {
  auto status = validate_manifests(manifests);
  if (!status.ok()) return status;
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  if (capacity_plan_instance.engine_epoch() != manifests[0].engine_epoch ||
      capacity_plan_instance.worker_generation() !=
          manifests[0].worker_generation ||
      capacity_plan_instance.world_size() != manifests.size() ||
      capacity_plan_instance.manifest_root() != *manifest_root ||
      capacity_plan_instance.resource_plan_root() !=
          preflight.resource_plan_root() ||
      preflight.engine_epoch() != manifests[0].engine_epoch ||
      preflight.worker_generation() != manifests[0].worker_generation ||
      preflight.world_size() != manifests.size() ||
      preflight.manifest_root() != *manifest_root) {
    return Status::FailedPrecondition(
        "DeepSeek rank spawn preflight differs from manifests");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-spawn-authorization:v1", 6);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, manifests[0].engine_epoch);
  if (status.ok())
    status = builder->add_u64(2, manifests[0].worker_generation);
  if (status.ok())
    status = builder->add_u32(3, static_cast<std::uint32_t>(manifests.size()));
  if (status.ok()) status = builder->add_hash(4, *manifest_root);
  if (status.ok())
    status = builder->add_hash(5, capacity_plan_instance.instance_root());
  if (status.ok())
    status = builder->add_hash(6, preflight.receipt_root());
  if (!status.ok()) return status;
  auto authorization_root = builder->finalize();
  if (!authorization_root.ok()) return authorization_root.status();
  auto capacity_authority_anchor =
      std::move(capacity_plan_instance.admission_anchor_);
  if (capacity_authority_anchor == nullptr ||
      !capacity_authority_anchor->production_ready()) {
    return Status::FailedPrecondition(
        "DeepSeek rank capacity authority lease is not retained");
  }
  return DeepSeekRankSpawnAuthorization(
      manifests[0].engine_epoch, manifests[0].worker_generation,
      static_cast<std::uint32_t>(manifests.size()), *manifest_root,
      capacity_plan_instance.instance_root(), preflight.resource_plan_root(),
      preflight.receipt_root(), *authorization_root,
      std::move(capacity_authority_anchor));
}

Result<DeepSeekRankProcessSupervisor> DeepSeekRankProcessSupervisor::Create(
    std::span<const DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnAuthorization authorization,
    DeepSeekRankProcessDriver& driver) {
  auto status = validate_manifests(manifests);
  if (!status.ok()) return status;
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  if (authorization.engine_epoch() != manifests[0].engine_epoch ||
      authorization.worker_generation() != manifests[0].worker_generation ||
      authorization.world_size() != manifests.size() ||
      authorization.manifest_root() != *manifest_root ||
      authorization.capacity_authority_anchor_ == nullptr ||
      !authorization.capacity_authority_anchor_->production_ready()) {
    return Status::FailedPrecondition(
        "DeepSeek rank spawn authorization differs from manifests");
  }
  return DeepSeekRankProcessSupervisor(
      std::vector<DeepSeekRankProcessManifest>(manifests.begin(), manifests.end()),
      std::move(authorization), driver);
}

DeepSeekRankProcessSupervisor::DeepSeekRankProcessSupervisor(
    std::vector<DeepSeekRankProcessManifest> manifests,
    DeepSeekRankSpawnAuthorization authorization,
    DeepSeekRankProcessDriver& driver) noexcept
    : manifests_(std::move(manifests)), handles_(manifests_.size()),
      ready_(manifests_.size(), false), exec_ready_(manifests_.size()),
      challenges_(manifests_.size(), 0),
      spawn_authorization_(std::move(authorization)),
      capacity_authority_anchor_(
          spawn_authorization_->capacity_authority_anchor_),
      spawn_authorization_root_(
          spawn_authorization_->authorization_root()),
      capacity_plan_instance_root_(
          spawn_authorization_->capacity_plan_instance_root()),
      manifest_root_(spawn_authorization_->manifest_root()),
      spawn_resource_plan_root_(
          spawn_authorization_->spawn_resource_plan_root()),
      driver_(&driver) {}

Status DeepSeekRankProcessSupervisor::fail(Status cause) noexcept {
  failed_ = true;
  for (const auto& handle : handles_) {
    if (handle) (void)driver_->terminate(*handle);
  }
  return cause.ok() ? Status::Internal("DeepSeek rank process generation failed")
                    : cause;
}

Status DeepSeekRankProcessSupervisor::launch() {
  if (launched_ || failed_ || !spawn_authorization_)
    return Status::FailedPrecondition("DeepSeek ranks cannot launch");
  spawn_authorization_.reset();
  launched_ = true;
  std::set<std::uint64_t> process_identities;
  std::set<std::uint64_t> pidfd_identities;
  std::set<std::uint64_t> control_identities;
  for (std::size_t i = 0; i < manifests_.size(); ++i) {
    auto handle = driver_->spawn(manifests_[i]);
    if (!handle.ok() || handle->process_identity == 0 ||
        handle->pidfd_identity == 0 || handle->control_identity == 0 ||
        !process_identities.insert(handle->process_identity).second ||
        !pidfd_identities.insert(handle->pidfd_identity).second ||
        !control_identities.insert(handle->control_identity).second) {
      if (handle.ok()) (void)driver_->terminate(*handle);
      return fail(handle.ok() ? Status::Internal("DeepSeek rank handle is invalid")
                              : handle.status());
    }
    handles_[i] = *handle;
  }
  return Status::Ok();
}

Status DeepSeekRankProcessSupervisor::dispatch_challenge(
    std::uint32_t rank, std::uint64_t controller_process_identity,
    std::uint64_t challenge_identity) {
  if (!launched_ || failed_ || rank >= manifests_.size() ||
      controller_process_identity == 0 || challenge_identity == 0 ||
      challenges_[rank] != 0 || ready_[rank] || !handles_[rank]) {
    return fail(Status::FailedPrecondition(
        "DeepSeek rank exec challenge is not bindable"));
  }
  auto status = driver_->send_challenge(
      *handles_[rank], {1, manifests_[rank], *handles_[rank],
                        controller_process_identity, challenge_identity});
  if (!status.ok()) return fail(status);
  challenges_[rank] = challenge_identity;
  return Status::Ok();
}

Status DeepSeekRankProcessSupervisor::dispatch_challenges(
    std::uint64_t controller_process_identity,
    std::span<const std::uint64_t> challenge_identities) {
  if (!launched_ || failed_ || controller_process_identity == 0 ||
      challenge_identities.size() != manifests_.size()) {
    return fail(Status::FailedPrecondition(
        "DeepSeek rank exec challenge set is not bindable"));
  }
  std::set<std::uint64_t> unique;
  for (std::size_t rank = 0; rank < manifests_.size(); ++rank) {
    if (challenge_identities[rank] == 0 ||
        !unique.insert(challenge_identities[rank]).second ||
        challenges_[rank] != 0 || ready_[rank] || !handles_[rank]) {
      return fail(Status::FailedPrecondition(
          "DeepSeek rank exec challenge set is not bindable"));
    }
  }
  for (std::size_t rank = 0; rank < manifests_.size(); ++rank) {
    const auto status = driver_->send_challenge(
        *handles_[rank],
        {1, manifests_[rank], *handles_[rank],
         controller_process_identity, challenge_identities[rank]});
    if (!status.ok()) return fail(status);
    challenges_[rank] = challenge_identities[rank];
  }
  return Status::Ok();
}

Status DeepSeekRankProcessSupervisor::poll_ready(std::uint32_t rank) {
  if (!launched_ || failed_ || rank >= manifests_.size() ||
      challenges_[rank] == 0 || ready_[rank] || !handles_[rank]) {
    return Status::FailedPrecondition("DeepSeek rank ready channel is not pollable");
  }
  auto result = driver_->poll_ready(*handles_[rank]);
  if (!result.ok()) return fail(result.status());
  if (!result->has_value())
    return Status::Unavailable("DeepSeek rank ready receipt is pending");
  return accept_ready((*result)->receipt, (*result)->challenge_identity);
}

Status DeepSeekRankProcessSupervisor::accept_ready(
    DeepSeekRankReadyReceipt receipt, std::uint64_t challenge_identity) {
  if (!launched_ || failed_ || receipt.rank >= manifests_.size() ||
      ready_[receipt.rank] || !handles_[receipt.rank] ||
      challenge_identity == 0 ||
      challenge_identity != challenges_[receipt.rank]) {
    return fail(Status::FailedPrecondition("DeepSeek rank ready receipt is not acceptable"));
  }
  const auto& m = manifests_[receipt.rank];
  const auto& h = *handles_[receipt.rank];
  if (receipt.engine_epoch != m.engine_epoch ||
      receipt.worker_generation != m.worker_generation ||
      receipt.physical_device_identity != m.physical_device_identity ||
      receipt.process_manifest_identity != m.process_manifest_identity ||
      receipt.process_identity != h.process_identity ||
      receipt.pidfd_identity != h.pidfd_identity ||
      receipt.control_identity != h.control_identity) {
    return fail(Status::FailedPrecondition("DeepSeek rank ready identity drifted"));
  }
  if (receipt.physical_device_uuid_commitment !=
          m.physical_device_uuid_commitment ||
      receipt.startup_device_ordinal != m.startup_device_ordinal) {
    return fail(Status::FailedPrecondition("DeepSeek rank ready identity drifted"));
  }
  if (receipt.startup_deadline_ns != m.startup_deadline_ns) {
    return fail(Status::FailedPrecondition("DeepSeek rank ready identity drifted"));
  }
  exec_ready_[receipt.rank] =
      DeepSeekRankExecReady{receipt, challenge_identity};
  ready_[receipt.rank] = true;
  return Status::Ok();
}

Status DeepSeekRankProcessSupervisor::advance_exec_startup(
    std::uint64_t now_ns) {
  if (!launched_ || failed_) {
    return Status::FailedPrecondition(
        "DeepSeek rank exec startup is not advanceable");
  }
  if (ready()) return Status::Ok();
  if (now_ns >= manifests_[0].startup_deadline_ns) return expire(now_ns);
  auto status = poll();
  if (!status.ok()) return status;
  bool pending = false;
  for (std::uint32_t rank = 0; rank < manifests_.size(); ++rank) {
    if (ready_[rank]) continue;
    if (challenges_[rank] == 0) {
      return fail(Status::FailedPrecondition(
          "DeepSeek rank exec challenge was not dispatched"));
    }
    status = poll_ready(rank);
    if (!status.ok() && status.code() == StatusCode::kUnavailable) {
      pending = true;
      continue;
    }
    if (!status.ok()) return status;
  }
  if (ready()) return Status::Ok();
  return pending
             ? Status::Unavailable("DeepSeek rank exec startup is pending")
             : fail(Status::Internal(
                   "DeepSeek rank exec startup made no progress"));
}

Status DeepSeekRankProcessSupervisor::abort_startup(Status cause) noexcept {
  if (!launched_ || failed_ || ready()) {
    return Status::FailedPrecondition(
        "DeepSeek rank exec startup is not abortable");
  }
  return fail(cause.ok()
                  ? Status::Internal(
                        "DeepSeek rank exec startup was aborted")
                  : cause);
}

Status DeepSeekRankProcessSupervisor::abort_post_exec(Status cause) noexcept {
  if (!launched_ || failed_ || !ready()) {
    return Status::FailedPrecondition(
        "DeepSeek rank post-exec generation is not abortable");
  }
  return fail(cause.ok()
                  ? Status::Internal(
                        "DeepSeek rank post-exec generation was aborted")
                  : cause);
}

Status DeepSeekRankProcessSupervisor::expire(std::uint64_t now_ns) {
  if (!launched_ || failed_)
    return Status::FailedPrecondition("DeepSeek rank startup is not expirable");
  if (ready()) return Status::Ok();
  if (now_ns < manifests_[0].startup_deadline_ns) return Status::Ok();
  return fail(Status::DeadlineExceeded("DeepSeek rank startup deadline expired"));
}

Status DeepSeekRankProcessSupervisor::poll() {
  if (!launched_ || failed_) return Status::FailedPrecondition("DeepSeek ranks are not observable");
  for (const auto& handle : handles_) {
    auto observation = driver_->observe(*handle);
    if (!observation.ok() || *observation != DeepSeekRankProcessObservation::kRunning)
      return fail(observation.ok() ? Status::Unavailable("DeepSeek rank exited")
                                   : observation.status());
  }
  return Status::Ok();
}

bool DeepSeekRankProcessSupervisor::ready() const noexcept {
  if (!launched_ || failed_) return false;
  for (const auto value : ready_) if (!value) return false;
  return true;
}

}  // namespace pih
