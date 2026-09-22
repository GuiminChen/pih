#include "pih/model/deepseek_rank_materialization_grant.h"

#include <limits>
#include <type_traits>
#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x50524958U;
constexpr std::uint16_t kGrantFrameType = 13;
constexpr std::uint16_t kGrantAckFrameType = 14;

static_assert(kDeepSeekRankMaterializationGrantFrameBytes ==
              8 + 16 + 8 + 40 + 20 + 8 + 448 + 32);
static_assert(kDeepSeekRankMaterializationGrantAckFrameBytes ==
              8 + 16 + 8 + 40 + 128 + 32);

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

bool valid_gpu_family(RuntimeProfileGpuFamily family) noexcept {
  return family == RuntimeProfileGpuFamily::kRtx4090D24GiB ||
         family == RuntimeProfileGpuFamily::kH100Pcie80GiB;
}

bool valid_residency(RuntimeProfileResidency residency) noexcept {
  return residency == RuntimeProfileResidency::kFullResident ||
         residency == RuntimeProfileResidency::kHostSpill;
}

Status validate_shape(
    const DeepSeekRankMaterializationGrantFields& fields) {
  if (fields.protocol_version != 2 || fields.engine_epoch == 0 ||
      fields.worker_generation == 0 || fields.world_size < 1 ||
      fields.world_size > 4 || fields.rank >= fields.world_size ||
      fields.process_manifest_identity == 0 ||
      fields.process_identity == 0 || fields.pidfd_identity == 0 ||
      fields.control_identity == 0 || fields.challenge_identity == 0 ||
      fields.device_ordinal < 0 || !valid_gpu_family(fields.gpu_family) ||
      !valid_residency(fields.residency) || fields.deadline_ns == 0 ||
      !nonzero(fields.profile_envelope_root) ||
      !nonzero(fields.device_observation_root) ||
      !nonzero(fields.kernel_closure_root) ||
      !nonzero(fields.graph_snapshot_root) ||
      !nonzero(fields.capacity_plan_instance_root) ||
      !nonzero(fields.first_resource_seal_root) ||
      !nonzero(fields.metadata_transaction_root) ||
      !nonzero(fields.mapping_owner_root) ||
      !nonzero(fields.report_root) ||
      !nonzero(fields.post_mapping_receipt_root) ||
      !nonzero(fields.post_mapping_seal_root) ||
      !nonzero(fields.mapping_owner_set_root) ||
      !nonzero(fields.receipt_set_root) ||
      !nonzero(fields.allocation_authority_root)) {
    return Status::InvalidArgument(
        "DeepSeek rank materialization grant shape is invalid");
  }
  return Status::Ok();
}

Status validate_ack_shape(
    const DeepSeekRankMaterializationGrantAckFields& fields) {
  if (fields.protocol_version != 1 || fields.engine_epoch == 0 ||
      fields.worker_generation == 0 || fields.world_size < 1 ||
      fields.world_size > 4 || fields.rank >= fields.world_size ||
      fields.process_manifest_identity == 0 ||
      fields.process_identity == 0 || fields.pidfd_identity == 0 ||
      fields.control_identity == 0 || fields.challenge_identity == 0 ||
      !nonzero(fields.grant_root) || !nonzero(fields.report_root) ||
      !nonzero(fields.mapping_owner_root) ||
      !nonzero(fields.post_mapping_seal_root)) {
    return Status::InvalidArgument(
        "DeepSeek rank materialization grant ACK shape is invalid");
  }
  return Status::Ok();
}

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

}  // namespace

Result<Sha256Digest> compile_deepseek_rank_materialization_grant_root(
    const DeepSeekRankMaterializationGrantFields& fields) {
  auto status = validate_shape(fields);
  if (!status.ok()) return status;
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-materialization-grant:v2", 29);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, fields.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, fields.worker_generation);
  if (status.ok()) status = builder->add_u32(3, fields.world_size);
  if (status.ok()) status = builder->add_u32(4, fields.rank);
  if (status.ok()) {
    status = builder->add_u64(5, fields.process_manifest_identity);
  }
  if (status.ok()) status = builder->add_u64(6, fields.process_identity);
  if (status.ok()) status = builder->add_u64(7, fields.pidfd_identity);
  if (status.ok()) status = builder->add_u64(8, fields.control_identity);
  if (status.ok()) status = builder->add_u64(9, fields.challenge_identity);
  if (status.ok()) {
    status = builder->add_u32(
        10, static_cast<std::uint32_t>(fields.device_ordinal));
  }
  if (status.ok()) {
    status = builder->add_u32(
        11, static_cast<std::uint32_t>(fields.gpu_family));
  }
  if (status.ok()) {
    status = builder->add_u32(
        12, static_cast<std::uint32_t>(fields.residency));
  }
  if (status.ok()) {
    status = builder->add_u32(13, fields.production_eligible ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u32(14, fields.dspark_enabled ? 1U : 0U);
  }
  if (status.ok()) status = builder->add_u64(15, fields.deadline_ns);
  if (status.ok()) {
    status = builder->add_hash(16, fields.profile_envelope_root);
  }
  if (status.ok()) {
    status = builder->add_hash(17, fields.device_observation_root);
  }
  if (status.ok()) {
    status = builder->add_hash(18, fields.kernel_closure_root);
  }
  if (status.ok()) {
    status = builder->add_hash(19, fields.graph_snapshot_root);
  }
  if (status.ok()) {
    status = builder->add_hash(20, fields.capacity_plan_instance_root);
  }
  if (status.ok()) {
    status = builder->add_hash(21, fields.first_resource_seal_root);
  }
  if (status.ok()) {
    status = builder->add_hash(22, fields.metadata_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(23, fields.mapping_owner_root);
  if (status.ok()) status = builder->add_hash(24, fields.report_root);
  if (status.ok()) {
    status = builder->add_hash(25, fields.post_mapping_receipt_root);
  }
  if (status.ok()) {
    status = builder->add_hash(26, fields.post_mapping_seal_root);
  }
  if (status.ok()) {
    status = builder->add_hash(27, fields.mapping_owner_set_root);
  }
  if (status.ok()) status = builder->add_hash(28, fields.receipt_set_root);
  if (status.ok()) {
    status = builder->add_hash(29, fields.allocation_authority_root);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<std::array<std::byte,
                  kDeepSeekRankMaterializationGrantFrameBytes>>
encode_deepseek_rank_materialization_grant(
    const DeepSeekRankMaterializationGrantFields& fields) {
  auto root = compile_deepseek_rank_materialization_grant_root(fields);
  if (!root.ok()) return root.status();
  std::array<std::byte,
             kDeepSeekRankMaterializationGrantFrameBytes> output{};
  std::size_t offset = 0;
  put(output, offset, kMagic);
  put(output, offset, kGrantFrameType);
  put(output, offset, std::uint16_t{2});
  put(output, offset, fields.engine_epoch);
  put(output, offset, fields.worker_generation);
  put(output, offset, fields.world_size);
  put(output, offset, fields.rank);
  put(output, offset, fields.process_manifest_identity);
  put(output, offset, fields.process_identity);
  put(output, offset, fields.pidfd_identity);
  put(output, offset, fields.control_identity);
  put(output, offset, fields.challenge_identity);
  put(output, offset, fields.device_ordinal);
  put(output, offset, static_cast<std::uint32_t>(fields.gpu_family));
  put(output, offset, static_cast<std::uint32_t>(fields.residency));
  put(output, offset,
      static_cast<std::uint32_t>(fields.production_eligible ? 1U : 0U));
  put(output, offset,
      static_cast<std::uint32_t>(fields.dspark_enabled ? 1U : 0U));
  put(output, offset, fields.deadline_ns);
  put_digest(output, offset, fields.profile_envelope_root);
  put_digest(output, offset, fields.device_observation_root);
  put_digest(output, offset, fields.kernel_closure_root);
  put_digest(output, offset, fields.graph_snapshot_root);
  put_digest(output, offset, fields.capacity_plan_instance_root);
  put_digest(output, offset, fields.first_resource_seal_root);
  put_digest(output, offset, fields.metadata_transaction_root);
  put_digest(output, offset, fields.mapping_owner_root);
  put_digest(output, offset, fields.report_root);
  put_digest(output, offset, fields.post_mapping_receipt_root);
  put_digest(output, offset, fields.post_mapping_seal_root);
  put_digest(output, offset, fields.mapping_owner_set_root);
  put_digest(output, offset, fields.receipt_set_root);
  put_digest(output, offset, fields.allocation_authority_root);
  put_digest(output, offset, *root);
  return output;
}

Result<DeepSeekRankMaterializationGrantFields>
decode_deepseek_rank_materialization_grant(
    std::span<const std::byte> frame) {
  if (frame.size() != kDeepSeekRankMaterializationGrantFrameBytes) {
    return Status::InvalidArgument(
        "DeepSeek rank materialization grant frame size is invalid");
  }
  std::size_t offset = 0;
  if (get<std::uint32_t>(frame, offset) != kMagic ||
      get<std::uint16_t>(frame, offset) != kGrantFrameType ||
      get<std::uint16_t>(frame, offset) != 2) {
    return Status::InvalidArgument(
        "DeepSeek rank materialization grant frame header is invalid");
  }
  DeepSeekRankMaterializationGrantFields fields{};
  fields.protocol_version = 2;
  fields.engine_epoch = get<std::uint64_t>(frame, offset);
  fields.worker_generation = get<std::uint64_t>(frame, offset);
  fields.world_size = get<std::uint32_t>(frame, offset);
  fields.rank = get<std::uint32_t>(frame, offset);
  fields.process_manifest_identity = get<std::uint64_t>(frame, offset);
  fields.process_identity = get<std::uint64_t>(frame, offset);
  fields.pidfd_identity = get<std::uint64_t>(frame, offset);
  fields.control_identity = get<std::uint64_t>(frame, offset);
  fields.challenge_identity = get<std::uint64_t>(frame, offset);
  const auto device_ordinal = get<std::uint32_t>(frame, offset);
  const auto gpu_family = get<std::uint32_t>(frame, offset);
  const auto residency = get<std::uint32_t>(frame, offset);
  const auto production_eligible = get<std::uint32_t>(frame, offset);
  const auto dspark_enabled = get<std::uint32_t>(frame, offset);
  fields.deadline_ns = get<std::uint64_t>(frame, offset);
  fields.profile_envelope_root = get_digest(frame, offset);
  fields.device_observation_root = get_digest(frame, offset);
  fields.kernel_closure_root = get_digest(frame, offset);
  fields.graph_snapshot_root = get_digest(frame, offset);
  fields.capacity_plan_instance_root = get_digest(frame, offset);
  fields.first_resource_seal_root = get_digest(frame, offset);
  fields.metadata_transaction_root = get_digest(frame, offset);
  fields.mapping_owner_root = get_digest(frame, offset);
  fields.report_root = get_digest(frame, offset);
  fields.post_mapping_receipt_root = get_digest(frame, offset);
  fields.post_mapping_seal_root = get_digest(frame, offset);
  fields.mapping_owner_set_root = get_digest(frame, offset);
  fields.receipt_set_root = get_digest(frame, offset);
  fields.allocation_authority_root = get_digest(frame, offset);
  const auto encoded_root = get_digest(frame, offset);
  if (device_ordinal >
          static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
      gpu_family < static_cast<std::uint32_t>(
                       RuntimeProfileGpuFamily::kRtx4090D24GiB) ||
      gpu_family > static_cast<std::uint32_t>(
                       RuntimeProfileGpuFamily::kH100Pcie80GiB) ||
      residency < static_cast<std::uint32_t>(
                      RuntimeProfileResidency::kFullResident) ||
      residency > static_cast<std::uint32_t>(
                      RuntimeProfileResidency::kHostSpill) ||
      production_eligible > 1 || dspark_enabled > 1) {
    return Status::InvalidArgument(
        "DeepSeek rank materialization grant enum is invalid");
  }
  fields.device_ordinal = static_cast<std::int32_t>(device_ordinal);
  fields.gpu_family = static_cast<RuntimeProfileGpuFamily>(gpu_family);
  fields.residency = static_cast<RuntimeProfileResidency>(residency);
  fields.production_eligible = production_eligible == 1;
  fields.dspark_enabled = dspark_enabled == 1;
  auto root = compile_deepseek_rank_materialization_grant_root(fields);
  if (!root.ok()) return root.status();
  if (*root != encoded_root) {
    return Status::FailedPrecondition(
        "DeepSeek rank materialization grant root differs");
  }
  return fields;
}

Result<Sha256Digest> compile_deepseek_rank_materialization_grant_ack_root(
    const DeepSeekRankMaterializationGrantAckFields& fields) {
  auto status = validate_ack_shape(fields);
  if (!status.ok()) return status;
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-materialization-grant-ack:v1", 13);
  if (!builder.ok()) return builder.status();
  status = builder->add_u64(1, fields.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, fields.worker_generation);
  if (status.ok()) status = builder->add_u32(3, fields.world_size);
  if (status.ok()) status = builder->add_u32(4, fields.rank);
  if (status.ok()) {
    status = builder->add_u64(5, fields.process_manifest_identity);
  }
  if (status.ok()) status = builder->add_u64(6, fields.process_identity);
  if (status.ok()) status = builder->add_u64(7, fields.pidfd_identity);
  if (status.ok()) status = builder->add_u64(8, fields.control_identity);
  if (status.ok()) status = builder->add_u64(9, fields.challenge_identity);
  if (status.ok()) status = builder->add_hash(10, fields.grant_root);
  if (status.ok()) status = builder->add_hash(11, fields.report_root);
  if (status.ok()) {
    status = builder->add_hash(12, fields.mapping_owner_root);
  }
  if (status.ok()) {
    status = builder->add_hash(13, fields.post_mapping_seal_root);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<std::array<
    std::byte, kDeepSeekRankMaterializationGrantAckFrameBytes>>
encode_deepseek_rank_materialization_grant_ack(
    const DeepSeekRankMaterializationGrantAckFields& fields) {
  auto root = compile_deepseek_rank_materialization_grant_ack_root(fields);
  if (!root.ok()) return root.status();
  std::array<std::byte,
             kDeepSeekRankMaterializationGrantAckFrameBytes> output{};
  std::size_t offset = 0;
  put(output, offset, kMagic);
  put(output, offset, kGrantAckFrameType);
  put(output, offset, std::uint16_t{1});
  put(output, offset, fields.engine_epoch);
  put(output, offset, fields.worker_generation);
  put(output, offset, fields.world_size);
  put(output, offset, fields.rank);
  put(output, offset, fields.process_manifest_identity);
  put(output, offset, fields.process_identity);
  put(output, offset, fields.pidfd_identity);
  put(output, offset, fields.control_identity);
  put(output, offset, fields.challenge_identity);
  put_digest(output, offset, fields.grant_root);
  put_digest(output, offset, fields.report_root);
  put_digest(output, offset, fields.mapping_owner_root);
  put_digest(output, offset, fields.post_mapping_seal_root);
  put_digest(output, offset, *root);
  return output;
}

Result<DeepSeekRankMaterializationGrantAckFields>
decode_deepseek_rank_materialization_grant_ack(
    std::span<const std::byte> frame) {
  if (frame.size() != kDeepSeekRankMaterializationGrantAckFrameBytes) {
    return Status::InvalidArgument(
        "DeepSeek rank materialization grant ACK frame size is invalid");
  }
  std::size_t offset = 0;
  if (get<std::uint32_t>(frame, offset) != kMagic ||
      get<std::uint16_t>(frame, offset) != kGrantAckFrameType ||
      get<std::uint16_t>(frame, offset) != 1) {
    return Status::InvalidArgument(
        "DeepSeek rank materialization grant ACK header is invalid");
  }
  DeepSeekRankMaterializationGrantAckFields fields{};
  fields.protocol_version = 1;
  fields.engine_epoch = get<std::uint64_t>(frame, offset);
  fields.worker_generation = get<std::uint64_t>(frame, offset);
  fields.world_size = get<std::uint32_t>(frame, offset);
  fields.rank = get<std::uint32_t>(frame, offset);
  fields.process_manifest_identity = get<std::uint64_t>(frame, offset);
  fields.process_identity = get<std::uint64_t>(frame, offset);
  fields.pidfd_identity = get<std::uint64_t>(frame, offset);
  fields.control_identity = get<std::uint64_t>(frame, offset);
  fields.challenge_identity = get<std::uint64_t>(frame, offset);
  fields.grant_root = get_digest(frame, offset);
  fields.report_root = get_digest(frame, offset);
  fields.mapping_owner_root = get_digest(frame, offset);
  fields.post_mapping_seal_root = get_digest(frame, offset);
  const auto encoded_root = get_digest(frame, offset);
  auto root = compile_deepseek_rank_materialization_grant_ack_root(fields);
  if (!root.ok()) return root.status();
  if (*root != encoded_root) {
    return Status::FailedPrecondition(
        "DeepSeek rank materialization grant ACK root differs");
  }
  return fields;
}

Result<DeepSeekRankMaterializationGrantFields>
compile_deepseek_rank_materialization_grant(
    const RuntimeEngineAdmission& admission,
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankPostMappingResourceCoordinator& coordinator,
    std::uint32_t rank, std::uint64_t deadline_ns,
    const DeepSeekRankMaterializationAllocationAuthority&
        allocation_authority) {
  const auto* seal = coordinator.seal();
  const auto* report = coordinator.report(rank);
  const auto* receipt = coordinator.receipt(rank);
  const auto* handle = supervisor.process_handle(rank);
  const auto* ready = supervisor.exec_ready(rank);
  const auto& roots = admission.reference_roots();
  auto allocation_authority_root =
      compile_deepseek_rank_materialization_allocation_authority_root(
          allocation_authority);
  if (!allocation_authority_root.ok()) return allocation_authority_root.status();
  if (!admission.production_ready() ||
      !admission.evidence_projection_bound() ||
      admission.model() != RuntimeProfileModel::kDeepSeekV4Flash0731 ||
      admission.weight_format() != RuntimeProfileWeightFormat::kDeepSeekNative ||
      admission.device_ordinals().size() != manifests.size() ||
      manifests.empty() || manifests.size() > 4 || rank >= manifests.size() ||
      !supervisor.ready() || supervisor.failed() ||
      supervisor.capacity_admission() != &admission ||
      !coordinator.sealed() || seal == nullptr || report == nullptr ||
      receipt == nullptr || handle == nullptr || ready == nullptr ||
      deadline_ns <= coordinator.deadline_ns() ||
      !manifest_matches_ready(manifests[rank], *ready) ||
      handle->process_identity != ready->receipt.process_identity ||
      handle->pidfd_identity != ready->receipt.pidfd_identity ||
      handle->control_identity != ready->receipt.control_identity ||
      admission.device_ordinals()[rank] < 0 ||
      static_cast<std::uint32_t>(admission.device_ordinals()[rank]) !=
          manifests[rank].startup_device_ordinal ||
      admission.dspark_enabled() != seal->dspark_enabled() ||
      allocation_authority.device_ordinal != admission.device_ordinals()[rank] ||
      allocation_authority.residency != admission.residency() ||
      report->engine_epoch() != seal->engine_epoch() ||
      report->worker_generation() != seal->worker_generation() ||
      report->world_size() != seal->world_size() ||
      report->rank() != rank ||
      report->process_identity() != handle->process_identity ||
      receipt->engine_epoch() != seal->engine_epoch() ||
      receipt->worker_generation() != seal->worker_generation() ||
      receipt->world_size() != seal->world_size() ||
      receipt->rank() != rank ||
      receipt->process_identity() != handle->process_identity ||
      receipt->mapping_owner_root() != report->mapping_owner_root() ||
      receipt->capacity_plan_instance_root() !=
          supervisor.capacity_plan_instance_root() ||
      receipt->first_resource_seal_root() !=
          seal->first_resource_seal_root() ||
      receipt->metadata_transaction_root() !=
          seal->metadata_transaction_root() ||
      receipt->dspark_enabled() != seal->dspark_enabled() ||
      !nonzero(admission.envelope_root()) ||
      !nonzero(admission.device_observation_root()) ||
      !nonzero(roots.kernel_closure_root) ||
      !nonzero(admission.graph_snapshot_root())) {
    return Status::FailedPrecondition(
        "DeepSeek rank materialization grant antecedents differ");
  }
  DeepSeekRankMaterializationGrantFields fields{
      2,
      seal->engine_epoch(),
      seal->worker_generation(),
      seal->world_size(),
      rank,
      manifests[rank].process_manifest_identity,
      handle->process_identity,
      handle->pidfd_identity,
      handle->control_identity,
      ready->challenge_identity,
      admission.device_ordinals()[rank],
      admission.gpu_family(),
      admission.residency(),
      seal->production_eligible(),
      seal->dspark_enabled(),
      deadline_ns,
      admission.envelope_root(),
      admission.device_observation_root(),
      roots.kernel_closure_root,
      admission.graph_snapshot_root(),
      supervisor.capacity_plan_instance_root(),
      seal->first_resource_seal_root(),
      seal->metadata_transaction_root(),
      report->mapping_owner_root(),
      report->report_root(),
      receipt->receipt_root(),
      seal->seal_root(),
      seal->mapping_owner_set_root(),
      seal->receipt_set_root(),
      *allocation_authority_root};
  auto root = compile_deepseek_rank_materialization_grant_root(fields);
  if (!root.ok()) return root.status();
  return fields;
}

Result<DeepSeekRankMaterializationAdmission>
DeepSeekRankMaterializationAdmission::Accept(
    const DeepSeekRankProcessManifest& manifest,
    const DeepSeekRankExecReady& exec_ready,
    const DeepSeekRankArtifactMappingOwner& mapping_owner,
    const DeepSeekRankPostMappingResourceReport& report,
    DeepSeekRankMaterializationGrantFields grant) {
  auto grant_root = compile_deepseek_rank_materialization_grant_root(grant);
  if (!grant_root.ok()) return grant_root.status();
  const auto& ready = exec_ready.receipt;
  if (!manifest_matches_ready(manifest, exec_ready) ||
      grant.engine_epoch != manifest.engine_epoch ||
      grant.worker_generation != manifest.worker_generation ||
      grant.world_size != manifest.world_size || grant.rank != manifest.rank ||
      grant.process_manifest_identity != manifest.process_manifest_identity ||
      grant.process_identity != ready.process_identity ||
      grant.pidfd_identity != ready.pidfd_identity ||
      grant.control_identity != ready.control_identity ||
      grant.challenge_identity != exec_ready.challenge_identity ||
      grant.device_ordinal < 0 ||
      static_cast<std::uint32_t>(grant.device_ordinal) !=
          manifest.startup_device_ordinal ||
      mapping_owner.engine_epoch() != manifest.engine_epoch ||
      mapping_owner.worker_generation() != manifest.worker_generation ||
      mapping_owner.world_size() != manifest.world_size ||
      mapping_owner.rank() != manifest.rank ||
      mapping_owner.mapping_owner_root() != grant.mapping_owner_root ||
      mapping_owner.metadata_transaction_root() !=
          grant.metadata_transaction_root ||
      mapping_owner.dspark_enabled() != grant.dspark_enabled ||
      report.engine_epoch() != manifest.engine_epoch ||
      report.worker_generation() != manifest.worker_generation ||
      report.world_size() != manifest.world_size ||
      report.rank() != manifest.rank ||
      report.process_identity() != ready.process_identity ||
      report.mapping_owner_root() != mapping_owner.mapping_owner_root() ||
      report.capacity_plan_instance_root() !=
          grant.capacity_plan_instance_root ||
      report.first_resource_seal_root() !=
          grant.first_resource_seal_root ||
      report.metadata_transaction_root() !=
          grant.metadata_transaction_root ||
      report.report_root() != grant.report_root ||
      grant.deadline_ns <= report.authority_deadline_ns() ||
      (grant.production_eligible &&
       (!mapping_owner.source_catalog_production_eligible() ||
        mapping_owner.immutability_mode() ==
            ArtifactImmutabilityMode::kUncalibrated))) {
    return Status::FailedPrecondition(
        "DeepSeek rank materialization grant differs from local owners");
  }
  return DeepSeekRankMaterializationAdmission(
      std::move(grant), *grant_root);
}

Result<DeepSeekRankMaterializationGrantAckFields>
compile_deepseek_rank_materialization_grant_ack(
    const DeepSeekRankMaterializationAdmission& admission) {
  const auto& grant = admission.fields();
  DeepSeekRankMaterializationGrantAckFields fields{
      1,
      grant.engine_epoch,
      grant.worker_generation,
      grant.world_size,
      grant.rank,
      grant.process_manifest_identity,
      grant.process_identity,
      grant.pidfd_identity,
      grant.control_identity,
      grant.challenge_identity,
      admission.grant_root(),
      grant.report_root,
      grant.mapping_owner_root,
      grant.post_mapping_seal_root};
  auto root = compile_deepseek_rank_materialization_grant_ack_root(fields);
  if (!root.ok()) return root.status();
  return fields;
}

}  // namespace pih
