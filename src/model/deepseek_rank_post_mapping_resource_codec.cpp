#include "pih/model/deepseek_rank_post_mapping_resource_codec.h"

#include <type_traits>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x50524958U;
constexpr std::uint16_t kObservationType = 11;
constexpr std::uint16_t kAuthorityType = 12;

static_assert(kDeepSeekRankPostMappingResourceObservationFrameBytes ==
              8 + 16 + 4 + 16 + 64 + 160 + 8 + 12 + 64 + 4 + 32);
static_assert(kDeepSeekRankPostMappingResourceAuthorityFrameBytes ==
              8 + 16 + 8 + 40 + 320 + 8 + 8 + 12 + 48 + 32);

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

bool valid_immutability(ArtifactImmutabilityMode mode) noexcept {
  switch (mode) {
    case ArtifactImmutabilityMode::kUncalibrated:
    case ArtifactImmutabilityMode::kFsVerity:
    case ArtifactImmutabilityMode::kDmVeritySnapshot:
      return true;
  }
  return false;
}

template <class T, std::size_t N>
void put(std::array<std::byte, N>& output, std::size_t& offset, T value) {
  using U = std::make_unsigned_t<T>;
  auto bits = static_cast<U>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output[offset++] = static_cast<std::byte>(
        (bits >> (index * 8U)) & static_cast<U>(0xffU));
  }
}

template <class T>
T get(std::span<const std::byte> input, std::size_t& offset) {
  using U = std::make_unsigned_t<T>;
  U value = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    value |= static_cast<U>(
                 std::to_integer<unsigned>(input[offset++]))
             << (index * 8U);
  }
  return static_cast<T>(value);
}

template <std::size_t N>
void put_digest(std::array<std::byte, N>& output, std::size_t& offset,
                const Sha256Digest& value) {
  for (const auto byte : value.bytes) output[offset++] = byte;
}

Sha256Digest get_digest(std::span<const std::byte> input,
                        std::size_t& offset) {
  Sha256Digest result{};
  for (auto& byte : result.bytes) byte = input[offset++];
  return result;
}

template <std::size_t N>
void put_header(std::array<std::byte, N>& output, std::size_t& offset,
                std::uint16_t type) {
  put(output, offset, kMagic);
  put(output, offset, type);
  put(output, offset, std::uint16_t{1});
}

Status validate_header(std::span<const std::byte> input,
                       std::size_t expected_bytes,
                       std::uint16_t expected_type,
                       std::size_t& offset) {
  if (input.size() != expected_bytes) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping resource frame size is invalid");
  }
  if (get<std::uint32_t>(input, offset) != kMagic ||
      get<std::uint16_t>(input, offset) != expected_type ||
      get<std::uint16_t>(input, offset) != 1) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping resource frame header is invalid");
  }
  return Status::Ok();
}

Status validate_observation_shape(
    const DeepSeekRankPostMappingResourceObservation& value) {
  const auto& resources = value.resources;
  if (resources.engine_epoch == 0 || resources.worker_generation == 0 ||
      resources.process_identity == 0 || resources.challenge_identity == 0 ||
      !nonzero(resources.acknowledged_capacity_plan_instance_root) ||
      !nonzero(resources.acknowledged_os_resource_envelope_root) ||
      !nonzero(value.first_resource_seal_root) ||
      !nonzero(value.descriptor_transaction_root) ||
      !nonzero(value.metadata_transaction_root) ||
      !nonzero(value.metadata_root) || !nonzero(value.mapping_owner_root) ||
      value.mapped_interval_bytes == 0 ||
      !valid_immutability(value.immutability_mode) ||
      resources.task_count == 0 || resources.open_fd_count == 0 ||
      resources.vma_count == 0 || resources.rlimit_nofile_soft == 0 ||
      resources.rlimit_nofile_hard < resources.rlimit_nofile_soft ||
      resources.rlimit_nofile_hard > resources.fs_nr_open ||
      resources.vm_max_map_count == 0 || !resources.non_dumpable) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping resource observation is invalid");
  }
  return Status::Ok();
}

Status validate_authority_shape(
    const DeepSeekRankPostMappingResourceAuthority& value) {
  const auto& plan = value.resource_plan;
  if (value.protocol_version != 1 || value.engine_epoch == 0 ||
      value.worker_generation == 0 || value.world_size < 1 ||
      value.world_size > 4 || value.rank >= value.world_size ||
      value.process_manifest_identity == 0 || value.process_identity == 0 ||
      value.pidfd_identity == 0 || value.control_identity == 0 ||
      value.challenge_identity == 0 || !nonzero(value.manifest_root) ||
      !nonzero(value.spawn_resource_plan_root) ||
      !nonzero(value.capacity_plan_instance_root) ||
      !nonzero(value.os_resource_envelope_root) ||
      !nonzero(value.first_resource_seal_root) ||
      !nonzero(value.first_resource_plan_set_root) ||
      !nonzero(value.descriptor_transaction_root) ||
      !nonzero(value.metadata_transaction_root) ||
      !nonzero(value.metadata_root) ||
      !nonzero(value.expected_mapping_owner_root) ||
      value.deadline_ns == 0 || value.expected_mapped_interval_bytes == 0 ||
      !valid_immutability(value.expected_immutability_mode) ||
      plan.worker_task_peak == 0 || plan.worker_fd_peak == 0 ||
      plan.worker_vma_peak == 0 || plan.task_emergency_reserve == 0 ||
      plan.fd_emergency_reserve == 0 || plan.vma_emergency_reserve == 0 ||
      plan.os_resource_envelope_root != value.os_resource_envelope_root) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping resource authority is invalid");
  }
  return Status::Ok();
}

}  // namespace

Result<Sha256Digest>
compile_deepseek_rank_post_mapping_resource_observation_frame_root(
    const DeepSeekRankPostMappingResourceObservation& value) {
  auto status = validate_observation_shape(value);
  if (!status.ok()) return status;
  const auto& resources = value.resources;
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-mapping-resource-observation-frame:v1",
      25);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, resources.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, resources.worker_generation);
  if (status.ok()) status = builder->add_u32(3, resources.rank);
  if (status.ok()) status = builder->add_u64(4, resources.process_identity);
  if (status.ok()) status = builder->add_u64(5, resources.challenge_identity);
  if (status.ok()) {
    status = builder->add_hash(
        6, resources.acknowledged_capacity_plan_instance_root);
  }
  if (status.ok()) {
    status = builder->add_hash(
        7, resources.acknowledged_os_resource_envelope_root);
  }
  if (status.ok()) status = builder->add_hash(8, value.first_resource_seal_root);
  if (status.ok()) status = builder->add_hash(9, value.descriptor_transaction_root);
  if (status.ok()) status = builder->add_hash(10, value.metadata_transaction_root);
  if (status.ok()) status = builder->add_hash(11, value.metadata_root);
  if (status.ok()) status = builder->add_hash(12, value.mapping_owner_root);
  if (status.ok()) status = builder->add_u64(13, value.mapped_interval_bytes);
  if (status.ok()) {
    status = builder->add_u32(
        14, static_cast<std::uint32_t>(value.immutability_mode));
  }
  if (status.ok()) {
    status = builder->add_u32(
        15, value.source_catalog_production_eligible ? 1U : 0U);
  }
  if (status.ok()) status = builder->add_u32(16, value.dspark_enabled ? 1U : 0U);
  if (status.ok()) status = builder->add_u64(17, resources.task_count);
  if (status.ok()) status = builder->add_u64(18, resources.open_fd_count);
  if (status.ok()) {
    status = builder->add_u64(19, resources.scm_rights_inflight_fd_count);
  }
  if (status.ok()) status = builder->add_u64(20, resources.vma_count);
  if (status.ok()) status = builder->add_u64(21, resources.rlimit_nofile_soft);
  if (status.ok()) status = builder->add_u64(22, resources.rlimit_nofile_hard);
  if (status.ok()) status = builder->add_u64(23, resources.fs_nr_open);
  if (status.ok()) status = builder->add_u64(24, resources.vm_max_map_count);
  if (status.ok()) {
    status = builder->add_u32(25, resources.non_dumpable ? 1U : 0U);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest>
compile_deepseek_rank_post_mapping_resource_authority_frame_root(
    const DeepSeekRankPostMappingResourceAuthority& value) {
  auto status = validate_authority_shape(value);
  if (!status.ok()) return status;
  const auto& plan = value.resource_plan;
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-post-mapping-resource-authority-frame:v1",
      30);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, value.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, value.worker_generation);
  if (status.ok()) status = builder->add_u32(3, value.world_size);
  if (status.ok()) status = builder->add_u32(4, value.rank);
  if (status.ok()) status = builder->add_u64(5, value.process_manifest_identity);
  if (status.ok()) status = builder->add_u64(6, value.process_identity);
  if (status.ok()) status = builder->add_u64(7, value.pidfd_identity);
  if (status.ok()) status = builder->add_u64(8, value.control_identity);
  if (status.ok()) status = builder->add_u64(9, value.challenge_identity);
  if (status.ok()) status = builder->add_hash(10, value.manifest_root);
  if (status.ok()) status = builder->add_hash(11, value.spawn_resource_plan_root);
  if (status.ok()) status = builder->add_hash(12, value.capacity_plan_instance_root);
  if (status.ok()) status = builder->add_hash(13, value.os_resource_envelope_root);
  if (status.ok()) status = builder->add_hash(14, value.first_resource_seal_root);
  if (status.ok()) {
    status = builder->add_hash(15, value.first_resource_plan_set_root);
  }
  if (status.ok()) status = builder->add_hash(16, value.descriptor_transaction_root);
  if (status.ok()) status = builder->add_hash(17, value.metadata_transaction_root);
  if (status.ok()) status = builder->add_hash(18, value.metadata_root);
  if (status.ok()) status = builder->add_hash(19, value.expected_mapping_owner_root);
  if (status.ok()) status = builder->add_u64(20, value.deadline_ns);
  if (status.ok()) {
    status = builder->add_u64(21, value.expected_mapped_interval_bytes);
  }
  if (status.ok()) {
    status = builder->add_u32(
        22, static_cast<std::uint32_t>(value.expected_immutability_mode));
  }
  if (status.ok()) {
    status = builder->add_u32(
        23, value.expected_source_catalog_production_eligible ? 1U : 0U);
  }
  if (status.ok()) status = builder->add_u32(24, value.dspark_enabled ? 1U : 0U);
  if (status.ok()) status = builder->add_u64(25, plan.worker_task_peak);
  if (status.ok()) status = builder->add_u64(26, plan.worker_fd_peak);
  if (status.ok()) status = builder->add_u64(27, plan.worker_vma_peak);
  if (status.ok()) status = builder->add_u64(28, plan.task_emergency_reserve);
  if (status.ok()) status = builder->add_u64(29, plan.fd_emergency_reserve);
  if (status.ok()) status = builder->add_u64(30, plan.vma_emergency_reserve);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<std::array<
    std::byte, kDeepSeekRankPostMappingResourceObservationFrameBytes>>
encode_deepseek_rank_post_mapping_resource_observation(
    const DeepSeekRankPostMappingResourceObservation& value) {
  auto root =
      compile_deepseek_rank_post_mapping_resource_observation_frame_root(
          value);
  if (!root.ok()) return root.status();
  std::array<std::byte,
             kDeepSeekRankPostMappingResourceObservationFrameBytes> output{};
  std::size_t offset = 0;
  put_header(output, offset, kObservationType);
  const auto& resources = value.resources;
  put(output, offset, resources.engine_epoch);
  put(output, offset, resources.worker_generation);
  put(output, offset, resources.rank);
  put(output, offset, resources.process_identity);
  put(output, offset, resources.challenge_identity);
  put_digest(output, offset,
             resources.acknowledged_capacity_plan_instance_root);
  put_digest(output, offset,
             resources.acknowledged_os_resource_envelope_root);
  put_digest(output, offset, value.first_resource_seal_root);
  put_digest(output, offset, value.descriptor_transaction_root);
  put_digest(output, offset, value.metadata_transaction_root);
  put_digest(output, offset, value.metadata_root);
  put_digest(output, offset, value.mapping_owner_root);
  put(output, offset, value.mapped_interval_bytes);
  put(output, offset,
      static_cast<std::uint32_t>(value.immutability_mode));
  put(output, offset, static_cast<std::uint32_t>(
                          value.source_catalog_production_eligible ? 1U : 0U));
  put(output, offset,
      static_cast<std::uint32_t>(value.dspark_enabled ? 1U : 0U));
  put(output, offset, resources.task_count);
  put(output, offset, resources.open_fd_count);
  put(output, offset, resources.scm_rights_inflight_fd_count);
  put(output, offset, resources.vma_count);
  put(output, offset, resources.rlimit_nofile_soft);
  put(output, offset, resources.rlimit_nofile_hard);
  put(output, offset, resources.fs_nr_open);
  put(output, offset, resources.vm_max_map_count);
  put(output, offset,
      static_cast<std::uint32_t>(resources.non_dumpable ? 1U : 0U));
  put_digest(output, offset, *root);
  return output;
}

Result<DeepSeekRankPostMappingResourceObservation>
decode_deepseek_rank_post_mapping_resource_observation(
    std::span<const std::byte> input) {
  std::size_t offset = 0;
  auto status = validate_header(
      input, kDeepSeekRankPostMappingResourceObservationFrameBytes,
      kObservationType, offset);
  if (!status.ok()) return status;
  DeepSeekRankPostMappingResourceObservation value{};
  auto& resources = value.resources;
  resources.engine_epoch = get<std::uint64_t>(input, offset);
  resources.worker_generation = get<std::uint64_t>(input, offset);
  resources.rank = get<std::uint32_t>(input, offset);
  resources.process_identity = get<std::uint64_t>(input, offset);
  resources.challenge_identity = get<std::uint64_t>(input, offset);
  resources.acknowledged_capacity_plan_instance_root =
      get_digest(input, offset);
  resources.acknowledged_os_resource_envelope_root =
      get_digest(input, offset);
  value.first_resource_seal_root = get_digest(input, offset);
  value.descriptor_transaction_root = get_digest(input, offset);
  value.metadata_transaction_root = get_digest(input, offset);
  value.metadata_root = get_digest(input, offset);
  value.mapping_owner_root = get_digest(input, offset);
  value.mapped_interval_bytes = get<std::uint64_t>(input, offset);
  const auto immutability = get<std::uint32_t>(input, offset);
  const auto production_eligible = get<std::uint32_t>(input, offset);
  const auto dspark = get<std::uint32_t>(input, offset);
  resources.task_count = get<std::uint64_t>(input, offset);
  resources.open_fd_count = get<std::uint64_t>(input, offset);
  resources.scm_rights_inflight_fd_count =
      get<std::uint64_t>(input, offset);
  resources.vma_count = get<std::uint64_t>(input, offset);
  resources.rlimit_nofile_soft = get<std::uint64_t>(input, offset);
  resources.rlimit_nofile_hard = get<std::uint64_t>(input, offset);
  resources.fs_nr_open = get<std::uint64_t>(input, offset);
  resources.vm_max_map_count = get<std::uint64_t>(input, offset);
  const auto non_dumpable = get<std::uint32_t>(input, offset);
  const auto encoded_root = get_digest(input, offset);
  if (immutability > static_cast<std::uint32_t>(
                         ArtifactImmutabilityMode::kDmVeritySnapshot) ||
      production_eligible > 1 || dspark > 1 || non_dumpable > 1) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping observation enum is invalid");
  }
  value.immutability_mode =
      static_cast<ArtifactImmutabilityMode>(immutability);
  value.source_catalog_production_eligible = production_eligible == 1;
  value.dspark_enabled = dspark == 1;
  resources.non_dumpable = non_dumpable == 1;
  auto root =
      compile_deepseek_rank_post_mapping_resource_observation_frame_root(
          value);
  if (!root.ok()) return root.status();
  if (*root != encoded_root) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping observation root differs");
  }
  return value;
}

Result<std::array<
    std::byte, kDeepSeekRankPostMappingResourceAuthorityFrameBytes>>
encode_deepseek_rank_post_mapping_resource_authority(
    const DeepSeekRankPostMappingResourceAuthority& value) {
  auto root =
      compile_deepseek_rank_post_mapping_resource_authority_frame_root(value);
  if (!root.ok()) return root.status();
  std::array<std::byte,
             kDeepSeekRankPostMappingResourceAuthorityFrameBytes> output{};
  std::size_t offset = 0;
  put_header(output, offset, kAuthorityType);
  put(output, offset, value.engine_epoch);
  put(output, offset, value.worker_generation);
  put(output, offset, value.world_size);
  put(output, offset, value.rank);
  put(output, offset, value.process_manifest_identity);
  put(output, offset, value.process_identity);
  put(output, offset, value.pidfd_identity);
  put(output, offset, value.control_identity);
  put(output, offset, value.challenge_identity);
  put_digest(output, offset, value.manifest_root);
  put_digest(output, offset, value.spawn_resource_plan_root);
  put_digest(output, offset, value.capacity_plan_instance_root);
  put_digest(output, offset, value.os_resource_envelope_root);
  put_digest(output, offset, value.first_resource_seal_root);
  put_digest(output, offset, value.first_resource_plan_set_root);
  put_digest(output, offset, value.descriptor_transaction_root);
  put_digest(output, offset, value.metadata_transaction_root);
  put_digest(output, offset, value.metadata_root);
  put_digest(output, offset, value.expected_mapping_owner_root);
  put(output, offset, value.deadline_ns);
  put(output, offset, value.expected_mapped_interval_bytes);
  put(output, offset,
      static_cast<std::uint32_t>(value.expected_immutability_mode));
  put(output, offset,
      static_cast<std::uint32_t>(
          value.expected_source_catalog_production_eligible ? 1U : 0U));
  put(output, offset,
      static_cast<std::uint32_t>(value.dspark_enabled ? 1U : 0U));
  put(output, offset, value.resource_plan.worker_task_peak);
  put(output, offset, value.resource_plan.worker_fd_peak);
  put(output, offset, value.resource_plan.worker_vma_peak);
  put(output, offset, value.resource_plan.task_emergency_reserve);
  put(output, offset, value.resource_plan.fd_emergency_reserve);
  put(output, offset, value.resource_plan.vma_emergency_reserve);
  put_digest(output, offset, *root);
  return output;
}

Result<DeepSeekRankPostMappingResourceAuthority>
decode_deepseek_rank_post_mapping_resource_authority(
    std::span<const std::byte> input) {
  std::size_t offset = 0;
  auto status = validate_header(
      input, kDeepSeekRankPostMappingResourceAuthorityFrameBytes,
      kAuthorityType, offset);
  if (!status.ok()) return status;
  DeepSeekRankPostMappingResourceAuthority value{};
  value.protocol_version = 1;
  value.engine_epoch = get<std::uint64_t>(input, offset);
  value.worker_generation = get<std::uint64_t>(input, offset);
  value.world_size = get<std::uint32_t>(input, offset);
  value.rank = get<std::uint32_t>(input, offset);
  value.process_manifest_identity = get<std::uint64_t>(input, offset);
  value.process_identity = get<std::uint64_t>(input, offset);
  value.pidfd_identity = get<std::uint64_t>(input, offset);
  value.control_identity = get<std::uint64_t>(input, offset);
  value.challenge_identity = get<std::uint64_t>(input, offset);
  value.manifest_root = get_digest(input, offset);
  value.spawn_resource_plan_root = get_digest(input, offset);
  value.capacity_plan_instance_root = get_digest(input, offset);
  value.os_resource_envelope_root = get_digest(input, offset);
  value.first_resource_seal_root = get_digest(input, offset);
  value.first_resource_plan_set_root = get_digest(input, offset);
  value.descriptor_transaction_root = get_digest(input, offset);
  value.metadata_transaction_root = get_digest(input, offset);
  value.metadata_root = get_digest(input, offset);
  value.expected_mapping_owner_root = get_digest(input, offset);
  value.deadline_ns = get<std::uint64_t>(input, offset);
  value.expected_mapped_interval_bytes =
      get<std::uint64_t>(input, offset);
  const auto immutability = get<std::uint32_t>(input, offset);
  const auto production_eligible = get<std::uint32_t>(input, offset);
  const auto dspark = get<std::uint32_t>(input, offset);
  value.resource_plan.worker_task_peak = get<std::uint64_t>(input, offset);
  value.resource_plan.worker_fd_peak = get<std::uint64_t>(input, offset);
  value.resource_plan.worker_vma_peak = get<std::uint64_t>(input, offset);
  value.resource_plan.task_emergency_reserve =
      get<std::uint64_t>(input, offset);
  value.resource_plan.fd_emergency_reserve =
      get<std::uint64_t>(input, offset);
  value.resource_plan.vma_emergency_reserve =
      get<std::uint64_t>(input, offset);
  value.resource_plan.os_resource_envelope_root =
      value.os_resource_envelope_root;
  const auto encoded_root = get_digest(input, offset);
  if (immutability > static_cast<std::uint32_t>(
                         ArtifactImmutabilityMode::kDmVeritySnapshot) ||
      production_eligible > 1 || dspark > 1) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping authority enum is invalid");
  }
  value.expected_immutability_mode =
      static_cast<ArtifactImmutabilityMode>(immutability);
  value.expected_source_catalog_production_eligible =
      production_eligible == 1;
  value.dspark_enabled = dspark == 1;
  auto root =
      compile_deepseek_rank_post_mapping_resource_authority_frame_root(value);
  if (!root.ok()) return root.status();
  if (*root != encoded_root) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping authority root differs");
  }
  return value;
}

}  // namespace pih
