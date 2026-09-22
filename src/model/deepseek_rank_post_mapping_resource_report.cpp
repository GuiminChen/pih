#include "pih/model/deepseek_rank_post_mapping_resource_report.h"

#include "pih/core/canonical_hash.h"

namespace pih {

DeepSeekRankPostMappingResourceReport::
    DeepSeekRankPostMappingResourceReport(
        std::uint64_t engine_epoch, std::uint64_t worker_generation,
        std::uint32_t world_size, std::uint32_t rank,
        std::uint64_t process_identity,
        std::uint64_t authority_deadline_ns,
        Sha256Digest mapping_owner_root,
        Sha256Digest capacity_plan_instance_root,
        Sha256Digest first_resource_seal_root,
        Sha256Digest metadata_transaction_root,
        Sha256Digest authority_frame_root,
        Sha256Digest observation_frame_root,
        Sha256Digest report_root) noexcept
    : engine_epoch_(engine_epoch),
      worker_generation_(worker_generation), world_size_(world_size),
      rank_(rank), process_identity_(process_identity),
      authority_deadline_ns_(authority_deadline_ns),
      mapping_owner_root_(mapping_owner_root),
      capacity_plan_instance_root_(capacity_plan_instance_root),
      first_resource_seal_root_(first_resource_seal_root),
      metadata_transaction_root_(metadata_transaction_root),
      authority_frame_root_(authority_frame_root),
      observation_frame_root_(observation_frame_root),
      report_root_(report_root) {}

Result<DeepSeekRankPostMappingResourceReport>
DeepSeekRankPostMappingResourceReport::Compile(
    const DeepSeekRankPostMappingResourceAuthority& authority,
    const DeepSeekRankPostMappingResourceObservation& observation) {
  const auto& resources = observation.resources;
  auto authority_root =
      compile_deepseek_rank_post_mapping_resource_authority_frame_root(
          authority);
  if (!authority_root.ok()) return authority_root.status();
  auto observation_root =
      compile_deepseek_rank_post_mapping_resource_observation_frame_root(
          observation);
  if (!observation_root.ok()) return observation_root.status();
  if (authority.engine_epoch != resources.engine_epoch ||
      authority.worker_generation != resources.worker_generation ||
      authority.rank != resources.rank ||
      authority.process_identity != resources.process_identity ||
      authority.challenge_identity != resources.challenge_identity ||
      authority.capacity_plan_instance_root !=
          resources.acknowledged_capacity_plan_instance_root ||
      authority.os_resource_envelope_root !=
          resources.acknowledged_os_resource_envelope_root ||
      authority.first_resource_seal_root !=
          observation.first_resource_seal_root ||
      authority.descriptor_transaction_root !=
          observation.descriptor_transaction_root ||
      authority.metadata_transaction_root !=
          observation.metadata_transaction_root ||
      authority.metadata_root != observation.metadata_root ||
      authority.expected_mapping_owner_root !=
          observation.mapping_owner_root ||
      authority.expected_mapped_interval_bytes !=
          observation.mapped_interval_bytes ||
      authority.expected_immutability_mode !=
          observation.immutability_mode ||
      authority.expected_source_catalog_production_eligible !=
          observation.source_catalog_production_eligible ||
      authority.dspark_enabled != observation.dspark_enabled ||
      resources.scm_rights_inflight_fd_count != 0) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping report authority differs from observation");
  }

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-mapping-resource-report:v1", 16);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, authority.engine_epoch);
  if (status.ok()) {
    status = builder->add_u64(2, authority.worker_generation);
  }
  if (status.ok()) status = builder->add_u32(3, authority.world_size);
  if (status.ok()) status = builder->add_u32(4, authority.rank);
  if (status.ok()) {
    status = builder->add_u64(5, authority.process_manifest_identity);
  }
  if (status.ok()) status = builder->add_u64(6, authority.process_identity);
  if (status.ok()) status = builder->add_u64(7, authority.pidfd_identity);
  if (status.ok()) status = builder->add_u64(8, authority.control_identity);
  if (status.ok()) status = builder->add_u64(9, authority.challenge_identity);
  if (status.ok()) {
    status = builder->add_hash(10, authority.capacity_plan_instance_root);
  }
  if (status.ok()) {
    status = builder->add_hash(11, authority.os_resource_envelope_root);
  }
  if (status.ok()) {
    status = builder->add_hash(12, authority.first_resource_seal_root);
  }
  if (status.ok()) {
    status = builder->add_hash(13, authority.metadata_transaction_root);
  }
  if (status.ok()) {
    status = builder->add_hash(14, observation.mapping_owner_root);
  }
  if (status.ok()) status = builder->add_hash(15, *authority_root);
  if (status.ok()) status = builder->add_hash(16, *observation_root);
  if (!status.ok()) return status;
  auto report_root = builder->finalize();
  if (!report_root.ok()) return report_root.status();
  return DeepSeekRankPostMappingResourceReport(
      authority.engine_epoch, authority.worker_generation,
      authority.world_size, authority.rank, authority.process_identity,
      authority.deadline_ns,
      observation.mapping_owner_root,
      authority.capacity_plan_instance_root,
      authority.first_resource_seal_root,
      authority.metadata_transaction_root, *authority_root, *observation_root,
      *report_root);
}

}  // namespace pih
