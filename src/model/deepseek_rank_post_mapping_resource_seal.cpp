#include "pih/model/deepseek_rank_post_mapping_resource_seal.h"

#include <array>
#include <set>
#include <vector>

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
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

Status validate_common_authority(
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> resource_plans,
    const DeepSeekRankPostExecResourceSeal& first_resource_seal,
    const DeepSeekRankArtifactMetadataTransferTransaction&
        metadata_transaction) {
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto spawn_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, spawn_plan);
  if (!spawn_root.ok()) return spawn_root.status();
  if (!supervisor.ready() || supervisor.failed() || manifests.empty() ||
      manifests.size() > 4 || resource_plans.size() != manifests.size() ||
      metadata_transaction.world_size() != manifests.size() ||
      !metadata_transaction.complete() || metadata_transaction.poisoned() ||
      !metadata_transaction.retains_descriptor_transaction() ||
      !nonzero(metadata_transaction.transaction_root()) ||
      *manifest_root != supervisor.manifest_root() ||
      *spawn_root != supervisor.spawn_resource_plan_root() ||
      first_resource_seal.engine_epoch() != manifests[0].engine_epoch ||
      first_resource_seal.worker_generation() !=
          manifests[0].worker_generation ||
      first_resource_seal.world_size() != manifests.size() ||
      first_resource_seal.capacity_plan_instance_root() !=
          supervisor.capacity_plan_instance_root() ||
      first_resource_seal.os_resource_envelope_root() !=
          resource_plans[0].os_resource_envelope_root ||
      !nonzero(first_resource_seal.seal_root())) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping resource authority is invalid");
  }
  auto plan_set_root =
      compile_deepseek_rank_post_exec_resource_plan_set_root(
          manifests, spawn_plan, resource_plans);
  if (!plan_set_root.ok()) return plan_set_root.status();
  if (*plan_set_root != first_resource_seal.plan_set_root()) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping resource plan set differs from first seal");
  }
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    if (manifests[rank].rank != rank ||
        manifests[rank].engine_epoch != manifests[0].engine_epoch ||
        manifests[rank].worker_generation !=
            manifests[0].worker_generation ||
        manifests[rank].world_size != manifests.size() ||
        supervisor.exec_ready(rank) == nullptr ||
        resource_plans[rank].os_resource_envelope_root !=
            resource_plans[0].os_resource_envelope_root) {
      return Status::FailedPrecondition(
          "DeepSeek post-mapping rank authority drifted");
    }
    auto plan_root = compile_deepseek_rank_post_exec_resource_plan_root(
        manifests, spawn_plan, rank, resource_plans[rank]);
    if (!plan_root.ok()) return plan_root.status();
  }
  return Status::Ok();
}

}  // namespace

DeepSeekRankPostMappingResourceReceipt::
    DeepSeekRankPostMappingResourceReceipt(
        std::uint64_t engine_epoch, std::uint64_t worker_generation,
        std::uint32_t world_size, std::uint32_t rank,
        std::uint64_t process_identity,
        Sha256Digest capacity_plan_instance_root,
        Sha256Digest os_resource_envelope_root,
        Sha256Digest first_resource_seal_root,
        Sha256Digest resource_plan_root,
        Sha256Digest descriptor_transaction_root,
        Sha256Digest metadata_transaction_root, Sha256Digest metadata_root,
        Sha256Digest mapping_owner_root,
        std::uint64_t mapped_interval_bytes,
        ArtifactImmutabilityMode immutability_mode,
        bool source_catalog_production_eligible, bool dspark_enabled,
        Sha256Digest receipt_root) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      world_size_(world_size), rank_(rank),
      process_identity_(process_identity),
      capacity_plan_instance_root_(capacity_plan_instance_root),
      os_resource_envelope_root_(os_resource_envelope_root),
      first_resource_seal_root_(first_resource_seal_root),
      resource_plan_root_(resource_plan_root),
      descriptor_transaction_root_(descriptor_transaction_root),
      metadata_transaction_root_(metadata_transaction_root),
      metadata_root_(metadata_root),
      mapping_owner_root_(mapping_owner_root),
      mapped_interval_bytes_(mapped_interval_bytes),
      immutability_mode_(immutability_mode),
      source_catalog_production_eligible_(
          source_catalog_production_eligible),
      dspark_enabled_(dspark_enabled), receipt_root_(receipt_root) {}

Result<DeepSeekRankPostMappingResourceReceipt>
DeepSeekRankPostMappingResourceReceipt::Compile(
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> resource_plans,
    const DeepSeekRankPostExecResourceSeal& first_resource_seal,
    const DeepSeekRankArtifactMetadataTransferTransaction&
        metadata_transaction,
    const DeepSeekRankPostMappingResourceObservation& observation) {
  if (observation.resources.rank >= manifests.size()) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping observation rank is invalid");
  }
  auto authority = validate_common_authority(
      supervisor, manifests, spawn_plan, resource_plans,
      first_resource_seal, metadata_transaction);
  if (!authority.ok()) return authority;
  const auto rank = observation.resources.rank;
  const auto& manifest = manifests[rank];
  const auto& resource_plan = resource_plans[rank];
  const auto* ready = supervisor.exec_ready(rank);
  auto plan_root = compile_deepseek_rank_post_exec_resource_plan_root(
      manifests, spawn_plan, rank, resource_plan);
  if (!plan_root.ok()) return plan_root.status();
  if (ready == nullptr || !same_exec_identity(manifest, *ready)) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping receipt authority is invalid");
  }
  const auto& resources = observation.resources;
  if (resources.engine_epoch != manifest.engine_epoch ||
      resources.worker_generation != manifest.worker_generation ||
      resources.rank != manifest.rank ||
      resources.process_identity != ready->receipt.process_identity ||
      resources.challenge_identity != ready->challenge_identity ||
      resources.acknowledged_capacity_plan_instance_root !=
          supervisor.capacity_plan_instance_root() ||
      resources.acknowledged_os_resource_envelope_root !=
          resource_plan.os_resource_envelope_root ||
      observation.first_resource_seal_root !=
          first_resource_seal.seal_root() ||
      observation.descriptor_transaction_root !=
          metadata_transaction.descriptor_transaction_root() ||
      observation.metadata_transaction_root !=
          metadata_transaction.transaction_root() ||
      observation.metadata_root !=
          metadata_transaction.metadata_root(rank) ||
      observation.mapping_owner_root !=
          metadata_transaction.expected_mapping_owner_root(rank) ||
      observation.mapped_interval_bytes !=
          metadata_transaction.expected_mapped_interval_bytes(rank) ||
      observation.mapped_interval_bytes == 0 ||
      observation.immutability_mode !=
          metadata_transaction.expected_immutability_mode(rank) ||
      observation.source_catalog_production_eligible !=
          metadata_transaction
              .expected_source_catalog_production_eligible(rank) ||
      observation.dspark_enabled != metadata_transaction.dspark_enabled() ||
      resources.task_count == 0 || resources.open_fd_count == 0 ||
      resources.vma_count == 0 ||
      resources.scm_rights_inflight_fd_count != 0 ||
      resources.rlimit_nofile_soft == 0 ||
      resources.rlimit_nofile_hard < resources.rlimit_nofile_soft ||
      resources.rlimit_nofile_hard > resources.fs_nr_open ||
      resources.vm_max_map_count == 0 || !resources.non_dumpable) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping resource observation is invalid");
  }
  auto fd_required = checked_add_u64(
      resource_plan.worker_fd_peak, resource_plan.fd_emergency_reserve);
  auto vma_required = checked_add_u64(
      resource_plan.worker_vma_peak, resource_plan.vma_emergency_reserve);
  auto task_required = checked_add_u64(
      resource_plan.worker_task_peak,
      resource_plan.task_emergency_reserve);
  if (!fd_required.ok() || !vma_required.ok() || !task_required.ok()) {
    return Status::ResourceExhausted(
        "DeepSeek post-mapping resource arithmetic overflowed");
  }
  if (resources.task_count > resource_plan.worker_task_peak ||
      resources.open_fd_count > resource_plan.worker_fd_peak ||
      resources.vma_count > resource_plan.worker_vma_peak ||
      *fd_required > resources.rlimit_nofile_soft ||
      *vma_required > resources.vm_max_map_count) {
    return Status::ResourceExhausted(
        "DeepSeek post-mapping resource envelope was exceeded");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-mapping-resource-receipt:v1", 34);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, manifest.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, manifest.worker_generation);
  if (status.ok()) status = builder->add_u32(3, manifest.world_size);
  if (status.ok()) status = builder->add_u32(4, manifest.rank);
  if (status.ok()) {
    status = builder->add_u64(5, manifest.process_manifest_identity);
  }
  if (status.ok()) {
    status = builder->add_u64(6, ready->receipt.process_identity);
  }
  if (status.ok()) {
    status = builder->add_u64(7, ready->receipt.pidfd_identity);
  }
  if (status.ok()) {
    status = builder->add_u64(8, ready->receipt.control_identity);
  }
  if (status.ok()) status = builder->add_u64(9, ready->challenge_identity);
  if (status.ok()) {
    status = builder->add_hash(
        10, supervisor.capacity_plan_instance_root());
  }
  if (status.ok()) {
    status = builder->add_hash(11, resource_plan.os_resource_envelope_root);
  }
  if (status.ok()) {
    status = builder->add_hash(12, first_resource_seal.seal_root());
  }
  if (status.ok()) status = builder->add_hash(13, *plan_root);
  if (status.ok()) {
    status = builder->add_hash(
        14, metadata_transaction.descriptor_transaction_root());
  }
  if (status.ok()) {
    status = builder->add_hash(15, metadata_transaction.transaction_root());
  }
  if (status.ok()) status = builder->add_hash(16, observation.metadata_root);
  if (status.ok()) {
    status = builder->add_hash(17, observation.mapping_owner_root);
  }
  if (status.ok()) {
    status = builder->add_u64(18, observation.mapped_interval_bytes);
  }
  if (status.ok()) {
    status = builder->add_u32(
        19, static_cast<std::uint32_t>(observation.immutability_mode));
  }
  if (status.ok()) {
    status = builder->add_u32(
        20, observation.source_catalog_production_eligible ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u32(21, observation.dspark_enabled ? 1U : 0U);
  }
  if (status.ok()) status = builder->add_u64(22, resources.task_count);
  if (status.ok()) status = builder->add_u64(23, resources.open_fd_count);
  if (status.ok()) {
    status = builder->add_u64(
        24, resources.scm_rights_inflight_fd_count);
  }
  if (status.ok()) status = builder->add_u64(25, resources.vma_count);
  if (status.ok()) {
    status = builder->add_u64(26, resources.rlimit_nofile_soft);
  }
  if (status.ok()) {
    status = builder->add_u64(27, resources.rlimit_nofile_hard);
  }
  if (status.ok()) status = builder->add_u64(28, resources.fs_nr_open);
  if (status.ok()) {
    status = builder->add_u64(29, resources.vm_max_map_count);
  }
  if (status.ok()) {
    status = builder->add_u32(30, resources.non_dumpable ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u64(31, resource_plan.worker_task_peak);
  }
  if (status.ok()) status = builder->add_u64(32, *task_required);
  if (status.ok()) status = builder->add_u64(33, *fd_required);
  if (status.ok()) status = builder->add_u64(34, *vma_required);
  if (!status.ok()) return status;
  auto receipt_root = builder->finalize();
  if (!receipt_root.ok()) return receipt_root.status();
  return DeepSeekRankPostMappingResourceReceipt(
      manifest.engine_epoch, manifest.worker_generation,
      manifest.world_size, manifest.rank, ready->receipt.process_identity,
      supervisor.capacity_plan_instance_root(),
      resource_plan.os_resource_envelope_root,
      first_resource_seal.seal_root(), *plan_root,
      metadata_transaction.descriptor_transaction_root(),
      metadata_transaction.transaction_root(), observation.metadata_root,
      observation.mapping_owner_root, observation.mapped_interval_bytes,
      observation.immutability_mode,
      observation.source_catalog_production_eligible,
      observation.dspark_enabled, *receipt_root);
}

DeepSeekRankPostMappingResourceSeal::DeepSeekRankPostMappingResourceSeal(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t world_size, Sha256Digest first_resource_seal_root,
    Sha256Digest metadata_transaction_root,
    Sha256Digest mapping_owner_set_root,
    Sha256Digest receipt_set_root, bool production_eligible,
    bool dspark_enabled, Sha256Digest seal_root) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      world_size_(world_size),
      first_resource_seal_root_(first_resource_seal_root),
      metadata_transaction_root_(metadata_transaction_root),
      mapping_owner_set_root_(mapping_owner_set_root),
      receipt_set_root_(receipt_set_root),
      production_eligible_(production_eligible),
      dspark_enabled_(dspark_enabled), seal_root_(seal_root) {}

Result<DeepSeekRankPostMappingResourceSeal>
DeepSeekRankPostMappingResourceSeal::Compile(
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> resource_plans,
    const DeepSeekRankPostExecResourceSeal& first_resource_seal,
    const DeepSeekRankArtifactMetadataTransferTransaction&
        metadata_transaction,
    std::span<const DeepSeekRankPostMappingResourceReceipt> receipts) {
  auto status = validate_common_authority(
      supervisor, manifests, spawn_plan, resource_plans,
      first_resource_seal, metadata_transaction);
  if (!status.ok()) return status;
  if (receipts.size() != manifests.size()) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping receipt set is incomplete");
  }
  auto manifest_root = compile_deepseek_rank_process_manifest_root(manifests);
  if (!manifest_root.ok()) return manifest_root.status();
  auto spawn_root = compile_deepseek_rank_spawn_resource_plan_root(
      manifests, spawn_plan);
  if (!spawn_root.ok()) return spawn_root.status();
  std::set<std::uint64_t> process_identities;
  std::set<std::array<std::byte, 32>> mapping_roots;
  std::set<std::array<std::byte, 32>> receipt_roots;
  std::vector<Sha256Digest> ordered_mapping_roots;
  std::vector<Sha256Digest> ordered_receipt_roots;
  ordered_mapping_roots.reserve(receipts.size());
  ordered_receipt_roots.reserve(receipts.size());
  bool production_eligible = true;
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    const auto& receipt = receipts[rank];
    const auto* ready = supervisor.exec_ready(rank);
    auto plan_root = compile_deepseek_rank_post_exec_resource_plan_root(
        manifests, spawn_plan, rank, resource_plans[rank]);
    if (!plan_root.ok()) return plan_root.status();
    if (ready == nullptr || receipt.engine_epoch() != manifests[rank].engine_epoch ||
        receipt.worker_generation() != manifests[rank].worker_generation ||
        receipt.world_size() != manifests.size() || receipt.rank() != rank ||
        receipt.process_identity() != ready->receipt.process_identity ||
        receipt.capacity_plan_instance_root() !=
            supervisor.capacity_plan_instance_root() ||
        receipt.os_resource_envelope_root() !=
            resource_plans[rank].os_resource_envelope_root ||
        receipt.first_resource_seal_root() !=
            first_resource_seal.seal_root() ||
        receipt.resource_plan_root() != *plan_root ||
        receipt.descriptor_transaction_root() !=
            metadata_transaction.descriptor_transaction_root() ||
        receipt.metadata_transaction_root() !=
            metadata_transaction.transaction_root() ||
        receipt.metadata_root() != metadata_transaction.metadata_root(rank) ||
        receipt.mapping_owner_root() !=
            metadata_transaction.expected_mapping_owner_root(rank) ||
        receipt.mapped_interval_bytes() !=
            metadata_transaction.expected_mapped_interval_bytes(rank) ||
        receipt.immutability_mode() !=
            metadata_transaction.expected_immutability_mode(rank) ||
        receipt.source_catalog_production_eligible() !=
            metadata_transaction
                .expected_source_catalog_production_eligible(rank) ||
        receipt.dspark_enabled() != metadata_transaction.dspark_enabled() ||
        !process_identities.insert(receipt.process_identity()).second ||
        !mapping_roots.insert(receipt.mapping_owner_root().bytes).second ||
        !receipt_roots.insert(receipt.receipt_root().bytes).second) {
      return Status::FailedPrecondition(
          "DeepSeek post-mapping receipt set drifted");
    }
    production_eligible =
        production_eligible &&
        receipt.source_catalog_production_eligible() &&
        receipt.immutability_mode() !=
            ArtifactImmutabilityMode::kUncalibrated;
    ordered_mapping_roots.push_back(receipt.mapping_owner_root());
    ordered_receipt_roots.push_back(receipt.receipt_root());
  }
  auto mapping_set = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-mapping-owner-set:v1",
      static_cast<std::uint32_t>(ordered_mapping_roots.size() + 1));
  if (!mapping_set.ok()) return mapping_set.status();
  status = mapping_set->add_u32(
      1, static_cast<std::uint32_t>(ordered_mapping_roots.size()));
  for (std::size_t rank = 0;
       status.ok() && rank < ordered_mapping_roots.size(); ++rank) {
    status = mapping_set->add_hash(
        static_cast<std::uint16_t>(rank + 2),
        ordered_mapping_roots[rank]);
  }
  if (!status.ok()) return status;
  auto mapping_set_root = mapping_set->finalize();
  if (!mapping_set_root.ok()) return mapping_set_root.status();
  auto receipt_set = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-mapping-receipt-set:v1",
      static_cast<std::uint32_t>(ordered_receipt_roots.size() + 1));
  if (!receipt_set.ok()) return receipt_set.status();
  status = receipt_set->add_u32(
      1, static_cast<std::uint32_t>(ordered_receipt_roots.size()));
  for (std::size_t rank = 0;
       status.ok() && rank < ordered_receipt_roots.size(); ++rank) {
    status = receipt_set->add_hash(
        static_cast<std::uint16_t>(rank + 2),
        ordered_receipt_roots[rank]);
  }
  if (!status.ok()) return status;
  auto receipt_set_root = receipt_set->finalize();
  if (!receipt_set_root.ok()) return receipt_set_root.status();
  auto seal = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-mapping-resource-seal:v1", 14);
  if (!seal.ok()) return seal.status();
  status = seal->add_u64(1, manifests[0].engine_epoch);
  if (status.ok()) status = seal->add_u64(2, manifests[0].worker_generation);
  if (status.ok()) {
    status = seal->add_u32(
        3, static_cast<std::uint32_t>(manifests.size()));
  }
  if (status.ok()) status = seal->add_hash(4, *manifest_root);
  if (status.ok()) status = seal->add_hash(5, *spawn_root);
  if (status.ok()) {
    status = seal->add_hash(6, supervisor.capacity_plan_instance_root());
  }
  if (status.ok()) {
    status = seal->add_hash(
        7, resource_plans[0].os_resource_envelope_root);
  }
  if (status.ok()) {
    status = seal->add_hash(8, first_resource_seal.seal_root());
  }
  if (status.ok()) {
    status = seal->add_hash(
        9, metadata_transaction.descriptor_transaction_root());
  }
  if (status.ok()) {
    status = seal->add_hash(10, metadata_transaction.transaction_root());
  }
  if (status.ok()) status = seal->add_hash(11, *mapping_set_root);
  if (status.ok()) status = seal->add_hash(12, *receipt_set_root);
  if (status.ok()) {
    status = seal->add_u32(13, production_eligible ? 1U : 0U);
  }
  if (status.ok()) {
    status = seal->add_u32(
        14, metadata_transaction.dspark_enabled() ? 1U : 0U);
  }
  if (!status.ok()) return status;
  auto seal_root = seal->finalize();
  if (!seal_root.ok()) return seal_root.status();
  return DeepSeekRankPostMappingResourceSeal(
      manifests[0].engine_epoch, manifests[0].worker_generation,
      static_cast<std::uint32_t>(manifests.size()),
      first_resource_seal.seal_root(),
      metadata_transaction.transaction_root(), *mapping_set_root,
      *receipt_set_root, production_eligible,
      metadata_transaction.dspark_enabled(), *seal_root);
}

}  // namespace pih
