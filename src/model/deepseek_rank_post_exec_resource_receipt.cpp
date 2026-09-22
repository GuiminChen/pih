#include "pih/model/deepseek_rank_post_exec_resource_receipt.h"

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

#include <set>

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) noexcept {
  for (const auto byte : value.bytes) {
    if (byte != std::byte{0}) return true;
  }
  return false;
}

Status validate_plan(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan, std::uint32_t rank,
    const DeepSeekRankPostExecResourcePlan& plan) {
  const auto spawn_status = DeepSeekRankSpawnPreflightReceipt::ValidatePlan(
      manifests, spawn_plan);
  if (!spawn_status.ok()) return spawn_status;
  if (rank >= manifests.size() ||
      plan.worker_task_peak == 0 ||
      plan.worker_fd_peak != spawn_plan.worker_fd_peak ||
      plan.worker_vma_peak != spawn_plan.worker_vma_peak ||
      plan.task_emergency_reserve == 0 ||
      plan.fd_emergency_reserve != spawn_plan.fd_emergency_reserve ||
      plan.vma_emergency_reserve != spawn_plan.vma_emergency_reserve ||
      !nonzero(plan.os_resource_envelope_root)) {
    return Status::InvalidArgument(
        "DeepSeek rank post-exec resource plan is invalid");
  }
  return Status::Ok();
}

bool same_exec_identity(const DeepSeekRankProcessManifest& manifest,
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

}  // namespace

Result<Sha256Digest> compile_deepseek_rank_post_exec_resource_plan_root(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan, std::uint32_t rank,
    const DeepSeekRankPostExecResourcePlan& plan) {
  auto status = validate_plan(manifests, spawn_plan, rank, plan);
  if (!status.ok()) return status;
  auto spawn_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, spawn_plan);
  if (!spawn_root.ok()) return spawn_root.status();
  const auto& manifest = manifests[rank];
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-exec-resource-plan:v1", 12);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, manifest.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, manifest.worker_generation);
  if (status.ok()) status = builder->add_u32(3, manifest.world_size);
  if (status.ok()) status = builder->add_u32(4, manifest.rank);
  if (status.ok()) status = builder->add_hash(5, *spawn_root);
  if (status.ok()) status = builder->add_u64(6, plan.worker_task_peak);
  if (status.ok()) status = builder->add_u64(7, plan.worker_fd_peak);
  if (status.ok()) status = builder->add_u64(8, plan.worker_vma_peak);
  if (status.ok()) status = builder->add_u64(9, plan.task_emergency_reserve);
  if (status.ok()) status = builder->add_u64(10, plan.fd_emergency_reserve);
  if (status.ok()) status = builder->add_u64(11, plan.vma_emergency_reserve);
  if (status.ok())
    status = builder->add_hash(12, plan.os_resource_envelope_root);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> compile_deepseek_rank_post_exec_resource_plan_set_root(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> plans) {
  if (manifests.empty() || plans.size() != manifests.size()) {
    return Status::InvalidArgument(
        "DeepSeek rank post-exec resource plan set is incomplete");
  }
  std::vector<Sha256Digest> ordered_plan_roots;
  ordered_plan_roots.reserve(plans.size());
  for (std::uint32_t rank = 0; rank < plans.size(); ++rank) {
    auto plan_root = compile_deepseek_rank_post_exec_resource_plan_root(
        manifests, spawn_plan, rank, plans[rank]);
    if (!plan_root.ok()) return plan_root.status();
    ordered_plan_roots.push_back(*plan_root);
  }
  auto plan_set = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-exec-plan-set:v1",
      static_cast<std::uint32_t>(ordered_plan_roots.size() + 1));
  if (!plan_set.ok()) return plan_set.status();
  auto status = plan_set->add_u32(
      1, static_cast<std::uint32_t>(ordered_plan_roots.size()));
  for (std::size_t rank = 0;
       status.ok() && rank < ordered_plan_roots.size(); ++rank) {
    status = plan_set->add_hash(
        static_cast<std::uint16_t>(rank + 2), ordered_plan_roots[rank]);
  }
  if (!status.ok()) return status;
  return plan_set->finalize();
}

DeepSeekRankPostExecResourceReceipt::DeepSeekRankPostExecResourceReceipt(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t rank, std::uint32_t world_size,
    std::uint64_t process_manifest_identity,
    std::uint64_t process_identity, std::uint64_t pidfd_identity,
    std::uint64_t control_identity, std::uint64_t challenge_identity,
    Sha256Digest capacity_plan_instance_root,
    Sha256Digest os_resource_envelope_root,
    Sha256Digest resource_plan_root,
    Sha256Digest receipt_root) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      rank_(rank), world_size_(world_size),
      process_manifest_identity_(process_manifest_identity),
      process_identity_(process_identity), pidfd_identity_(pidfd_identity),
      control_identity_(control_identity),
      challenge_identity_(challenge_identity),
      capacity_plan_instance_root_(capacity_plan_instance_root),
      os_resource_envelope_root_(os_resource_envelope_root),
      resource_plan_root_(resource_plan_root), receipt_root_(receipt_root) {}

Result<DeepSeekRankPostExecResourceReceipt>
DeepSeekRankPostExecResourceReceipt::Compile(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    const DeepSeekRankPostExecResourcePlan& plan,
    const DeepSeekRankExecReady& exec_ready,
    const DeepSeekRankPostExecResourceObservation& observation) {
  if (observation.rank >= manifests.size()) {
    return Status::InvalidArgument(
        "DeepSeek rank post-exec observation rank is invalid");
  }
  const auto& manifest = manifests[observation.rank];
  auto status = validate_plan(manifests, spawn_plan, observation.rank, plan);
  if (!status.ok()) return status;
  if (!same_exec_identity(manifest, exec_ready) ||
      observation.engine_epoch != manifest.engine_epoch ||
      observation.worker_generation != manifest.worker_generation ||
      observation.rank != manifest.rank ||
      observation.process_identity != exec_ready.receipt.process_identity ||
      observation.challenge_identity != exec_ready.challenge_identity ||
      !nonzero(observation.acknowledged_capacity_plan_instance_root) ||
      observation.acknowledged_os_resource_envelope_root !=
          plan.os_resource_envelope_root ||
      observation.task_count == 0 || observation.open_fd_count == 0 ||
      observation.vma_count == 0 ||
      observation.scm_rights_inflight_fd_count != 0 ||
      observation.rlimit_nofile_soft == 0 ||
      observation.rlimit_nofile_hard < observation.rlimit_nofile_soft ||
      observation.rlimit_nofile_hard > observation.fs_nr_open ||
      observation.vm_max_map_count == 0 || !observation.non_dumpable) {
    return Status::FailedPrecondition(
        "DeepSeek rank post-exec resource observation is invalid");
  }
  auto fd_required =
      checked_add_u64(plan.worker_fd_peak, plan.fd_emergency_reserve);
  auto vma_required =
      checked_add_u64(plan.worker_vma_peak, plan.vma_emergency_reserve);
  auto task_required =
      checked_add_u64(plan.worker_task_peak, plan.task_emergency_reserve);
  if (!fd_required.ok() || !vma_required.ok() || !task_required.ok()) {
    return Status::ResourceExhausted(
        "DeepSeek rank post-exec resource arithmetic overflowed");
  }
  if (observation.task_count > plan.worker_task_peak ||
      observation.open_fd_count > plan.worker_fd_peak ||
      observation.vma_count > plan.worker_vma_peak ||
      *fd_required > observation.rlimit_nofile_soft ||
      *vma_required > observation.vm_max_map_count) {
    return Status::ResourceExhausted(
        "DeepSeek rank post-exec resource envelope was exceeded");
  }
  auto plan_root = compile_deepseek_rank_post_exec_resource_plan_root(
      manifests, spawn_plan, observation.rank, plan);
  if (!plan_root.ok()) return plan_root.status();
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-exec-resource-receipt:v1", 25);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, manifest.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, manifest.worker_generation);
  if (status.ok()) status = builder->add_u32(3, manifest.world_size);
  if (status.ok()) status = builder->add_u32(4, manifest.rank);
  if (status.ok())
    status = builder->add_u64(5, manifest.process_manifest_identity);
  if (status.ok())
    status = builder->add_u64(6, exec_ready.receipt.process_identity);
  if (status.ok())
    status = builder->add_u64(7, exec_ready.receipt.pidfd_identity);
  if (status.ok())
    status = builder->add_u64(8, exec_ready.receipt.control_identity);
  if (status.ok()) status = builder->add_u64(9, exec_ready.challenge_identity);
  if (status.ok()) status = builder->add_hash(
      10, observation.acknowledged_capacity_plan_instance_root);
  if (status.ok())
    status = builder->add_hash(11, plan.os_resource_envelope_root);
  if (status.ok()) status = builder->add_hash(12, *plan_root);
  if (status.ok()) status = builder->add_u64(13, observation.task_count);
  if (status.ok()) status = builder->add_u64(14, observation.open_fd_count);
  if (status.ok()) status = builder->add_u64(
      15, observation.scm_rights_inflight_fd_count);
  if (status.ok()) status = builder->add_u64(16, observation.vma_count);
  if (status.ok())
    status = builder->add_u64(17, observation.rlimit_nofile_soft);
  if (status.ok())
    status = builder->add_u64(18, observation.rlimit_nofile_hard);
  if (status.ok()) status = builder->add_u64(19, observation.fs_nr_open);
  if (status.ok())
    status = builder->add_u64(20, observation.vm_max_map_count);
  if (status.ok())
    status = builder->add_u32(21, observation.non_dumpable ? 1U : 0U);
  if (status.ok()) status = builder->add_u64(22, plan.worker_task_peak);
  if (status.ok()) status = builder->add_u64(23, *task_required);
  if (status.ok()) status = builder->add_u64(24, *fd_required);
  if (status.ok()) status = builder->add_u64(25, *vma_required);
  if (!status.ok()) return status;
  auto receipt_root = builder->finalize();
  if (!receipt_root.ok()) return receipt_root.status();
  return DeepSeekRankPostExecResourceReceipt(
      manifest.engine_epoch, manifest.worker_generation, manifest.rank,
      manifest.world_size, manifest.process_manifest_identity,
      exec_ready.receipt.process_identity, exec_ready.receipt.pidfd_identity,
      exec_ready.receipt.control_identity, exec_ready.challenge_identity,
      observation.acknowledged_capacity_plan_instance_root,
      plan.os_resource_envelope_root, *plan_root, *receipt_root);
}

DeepSeekRankPostExecResourceSeal::DeepSeekRankPostExecResourceSeal(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t world_size, Sha256Digest capacity_plan_instance_root,
    Sha256Digest os_resource_envelope_root, Sha256Digest plan_set_root,
    Sha256Digest receipt_set_root,
    Sha256Digest seal_root) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      world_size_(world_size),
      capacity_plan_instance_root_(capacity_plan_instance_root),
      os_resource_envelope_root_(os_resource_envelope_root),
      plan_set_root_(plan_set_root),
      receipt_set_root_(receipt_set_root), seal_root_(seal_root) {}

Result<DeepSeekRankPostExecResourceSeal>
DeepSeekRankPostExecResourceSeal::Compile(
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> plans,
    std::span<const DeepSeekRankPostExecResourceReceipt> receipts) {
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto spawn_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, spawn_plan);
  if (!spawn_root.ok()) return spawn_root.status();
  if (!supervisor.ready() || supervisor.failed() ||
      manifests.empty() || plans.size() != manifests.size() ||
      receipts.size() != manifests.size() ||
      *manifest_root != supervisor.manifest_root() ||
      *spawn_root != supervisor.spawn_resource_plan_root() ||
      !nonzero(supervisor.capacity_plan_instance_root()) ||
      !nonzero(supervisor.spawn_authorization_root())) {
    return Status::FailedPrecondition(
        "DeepSeek rank post-exec resource set is incomplete");
  }
  const auto envelope_root = plans[0].os_resource_envelope_root;
  std::set<std::uint64_t> process_identities;
  std::set<std::array<std::byte, 32>> plan_roots;
  std::set<std::array<std::byte, 32>> receipt_roots;
  std::vector<Sha256Digest> ordered_receipt_roots;
  ordered_receipt_roots.reserve(manifests.size());
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    auto plan_root = compile_deepseek_rank_post_exec_resource_plan_root(
        manifests, spawn_plan, rank, plans[rank]);
    if (!plan_root.ok()) return plan_root.status();
    const auto* ready = supervisor.exec_ready(rank);
    const auto& receipt = receipts[rank];
    if (ready == nullptr ||
        plans[rank].os_resource_envelope_root != envelope_root ||
        receipt.engine_epoch() != manifests[rank].engine_epoch ||
        receipt.worker_generation() != manifests[rank].worker_generation ||
        receipt.rank() != rank || receipt.world_size() != manifests.size() ||
        receipt.process_manifest_identity() !=
            manifests[rank].process_manifest_identity ||
        receipt.process_identity() != ready->receipt.process_identity ||
        receipt.pidfd_identity() != ready->receipt.pidfd_identity ||
        receipt.control_identity() != ready->receipt.control_identity ||
        receipt.challenge_identity() != ready->challenge_identity ||
        receipt.capacity_plan_instance_root() !=
            supervisor.capacity_plan_instance_root() ||
        receipt.os_resource_envelope_root() != envelope_root ||
        receipt.resource_plan_root() != *plan_root ||
        !process_identities.insert(receipt.process_identity()).second ||
        !plan_roots.insert(plan_root->bytes).second ||
        !receipt_roots.insert(receipt.receipt_root().bytes).second) {
      return Status::FailedPrecondition(
          "DeepSeek rank post-exec resource receipt set drifted");
    }
    ordered_receipt_roots.push_back(receipt.receipt_root());
  }
  auto plan_set_root =
      compile_deepseek_rank_post_exec_resource_plan_set_root(
          manifests, spawn_plan, plans);
  if (!plan_set_root.ok()) return plan_set_root.status();
  auto receipt_set = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-exec-receipt-set:v1",
      static_cast<std::uint32_t>(ordered_receipt_roots.size() + 1));
  if (!receipt_set.ok()) return receipt_set.status();
  auto status = receipt_set->add_u32(
      1, static_cast<std::uint32_t>(ordered_receipt_roots.size()));
  for (std::size_t rank = 0;
       status.ok() && rank < ordered_receipt_roots.size(); ++rank) {
    status = receipt_set->add_hash(
        static_cast<std::uint16_t>(rank + 2), ordered_receipt_roots[rank]);
  }
  if (!status.ok()) return status;
  auto receipt_set_root = receipt_set->finalize();
  if (!receipt_set_root.ok()) return receipt_set_root.status();
  auto seal = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-exec-resource-seal:v1", 10);
  if (!seal.ok()) return seal.status();
  status = seal->add_u64(1, manifests[0].engine_epoch);
  if (status.ok()) status = seal->add_u64(2, manifests[0].worker_generation);
  if (status.ok())
    status = seal->add_u32(3, static_cast<std::uint32_t>(manifests.size()));
  if (status.ok()) status = seal->add_hash(4, *manifest_root);
  if (status.ok()) status = seal->add_hash(5, *spawn_root);
  if (status.ok())
    status = seal->add_hash(6, supervisor.capacity_plan_instance_root());
  if (status.ok())
    status = seal->add_hash(7, supervisor.spawn_authorization_root());
  if (status.ok()) status = seal->add_hash(8, envelope_root);
  if (status.ok()) status = seal->add_hash(9, *plan_set_root);
  if (status.ok()) status = seal->add_hash(10, *receipt_set_root);
  if (!status.ok()) return status;
  auto seal_root = seal->finalize();
  if (!seal_root.ok()) return seal_root.status();
  return DeepSeekRankPostExecResourceSeal(
      manifests[0].engine_epoch, manifests[0].worker_generation,
      static_cast<std::uint32_t>(manifests.size()),
      supervisor.capacity_plan_instance_root(), envelope_root,
      *plan_set_root, *receipt_set_root, *seal_root);
}

}  // namespace pih
