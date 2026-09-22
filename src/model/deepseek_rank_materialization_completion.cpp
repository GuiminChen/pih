#include "pih/model/deepseek_rank_materialization_completion.h"

#include <limits>
#include <type_traits>

#include "pih/core/canonical_hash.h"
#include "pih/model/deepseek_rank_engine_resources.h"

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x50524958U;
constexpr std::uint16_t kFrameType = 15;

static_assert(kDeepSeekRankMaterializationCompletionFrameBytes ==
              8 + 16 + 8 + 40 + 20 + 24 + 80 + 12 + 416 + 32);

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
    const DeepSeekRankMaterializationCompletionFields& fields) {
  const bool host_spill =
      fields.residency == RuntimeProfileResidency::kHostSpill;
  const bool pinned_shape =
      host_spill
          ? fields.pinned_staging_bytes > 0 &&
                fields.pinned_staging_allocation_generation > 0 &&
                fields.expert_slot_count >=
                    DeepSeekExpertPager::kTransferReservationWindow &&
                fields.staging_extent_count >=
                    DeepSeekExpertPager::kTransferReservationWindow &&
                fields.pager_transfer_reservations == 0 &&
                nonzero(fields.pinned_allocation_root)
          : fields.pinned_staging_bytes == 0 &&
                fields.pinned_staging_allocation_generation == 0 &&
                fields.expert_slot_count == 0 &&
                fields.staging_extent_count == 0 &&
                fields.pager_transfer_reservations == 0 &&
                !nonzero(fields.pinned_allocation_root);
  if (fields.protocol_version != 1 || fields.engine_epoch == 0 ||
      fields.worker_generation == 0 || fields.world_size < 1 ||
      fields.world_size > 4 || fields.rank >= fields.world_size ||
      fields.process_manifest_identity == 0 || fields.process_identity == 0 ||
      fields.pidfd_identity == 0 || fields.control_identity == 0 ||
      fields.challenge_identity == 0 || fields.device_ordinal < 0 ||
      !valid_gpu_family(fields.gpu_family) ||
      !valid_residency(fields.residency) ||
      fields.prefault_completed_monotonic_ns == 0 ||
      fields.completion_monotonic_ns <=
          fields.prefault_completed_monotonic_ns ||
      fields.deadline_ns <= fields.completion_monotonic_ns ||
      fields.mapped_interval_bytes == 0 ||
      fields.selected_page_union_bytes < fields.mapped_interval_bytes ||
      fields.selected_page_union_bytes == 0 ||
      (fields.selected_page_union_bytes %
       kDeepSeekArtifactPrefaultPageBytes) != 0 ||
      fields.resident_selected_page_bytes !=
          fields.selected_page_union_bytes ||
      fields.prefault_major_fault_count !=
          fields.completion_major_fault_count ||
      fields.fixed_weight_backing_bytes == 0 ||
      fields.fixed_weight_payload_bytes == 0 ||
      fields.fixed_weight_payload_bytes > fields.fixed_weight_backing_bytes ||
      fields.fixed_weight_allocation_generation == 0 || !pinned_shape ||
      !nonzero(fields.profile_envelope_root) ||
      !nonzero(fields.device_observation_root) ||
      !nonzero(fields.capacity_plan_instance_root) ||
      !nonzero(fields.post_mapping_seal_root) ||
      !nonzero(fields.metadata_transaction_root) ||
      !nonzero(fields.mapping_owner_root) || !nonzero(fields.grant_root) ||
      !nonzero(fields.prefault_layout_root) ||
      !nonzero(fields.prefault_receipt_root) ||
      !nonzero(fields.weight_layout_root) ||
      !nonzero(fields.weight_seal_root) ||
      !nonzero(fields.cuda_allocation_root)) {
    return Status::InvalidArgument(
        "DeepSeek rank materialization completion shape is invalid");
  }
  return Status::Ok();
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
    value |= static_cast<U>(std::to_integer<unsigned>(input[offset++]))
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

Status validate_deepseek_rank_materialization_completion_worker_identity(
    const DeepSeekRankMaterializationAdmission& admission,
    const DeepSeekRankExecReady& ready) {
  const auto& grant = admission.fields();
  if (admission.engine_epoch() != ready.receipt.engine_epoch ||
      admission.worker_generation() != ready.receipt.worker_generation ||
      admission.rank() != ready.receipt.rank ||
      grant.process_manifest_identity != ready.receipt.process_manifest_identity ||
      grant.process_identity != ready.receipt.process_identity ||
      grant.pidfd_identity != ready.receipt.pidfd_identity ||
      grant.control_identity != ready.receipt.control_identity ||
      grant.challenge_identity != ready.challenge_identity) {
    return Status::FailedPrecondition(
        "DeepSeek completion admission differs from worker identity");
  }
  return Status::Ok();
}

Result<Sha256Digest> compile_deepseek_rank_materialization_completion_root(
    const DeepSeekRankMaterializationCompletionFields& fields) {
  auto status = validate_shape(fields);
  if (!status.ok()) return status;
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-materialization-completion:v1", 43);
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
    status = builder->add_u32(11,
        static_cast<std::uint32_t>(fields.gpu_family));
  }
  if (status.ok()) {
    status = builder->add_u32(12,
        static_cast<std::uint32_t>(fields.residency));
  }
  if (status.ok()) {
    status = builder->add_u32(13, fields.production_eligible ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u32(14, fields.dspark_enabled ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u64(
        15, fields.prefault_completed_monotonic_ns);
  }
  if (status.ok()) {
    status = builder->add_u64(16, fields.completion_monotonic_ns);
  }
  if (status.ok()) status = builder->add_u64(17, fields.deadline_ns);
  if (status.ok()) status = builder->add_u64(18, fields.mapped_interval_bytes);
  if (status.ok()) {
    status = builder->add_u64(19, fields.selected_page_union_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(20, fields.resident_selected_page_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(21, fields.prefault_major_fault_count);
  }
  if (status.ok()) {
    status = builder->add_u64(22, fields.completion_major_fault_count);
  }
  if (status.ok()) {
    status = builder->add_u64(23, fields.fixed_weight_backing_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(24, fields.fixed_weight_payload_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(
        25, fields.fixed_weight_allocation_generation);
  }
  if (status.ok()) status = builder->add_u64(26, fields.pinned_staging_bytes);
  if (status.ok()) {
    status = builder->add_u64(
        27, fields.pinned_staging_allocation_generation);
  }
  if (status.ok()) status = builder->add_u32(28, fields.expert_slot_count);
  if (status.ok()) status = builder->add_u32(29, fields.staging_extent_count);
  if (status.ok()) {
    status = builder->add_u32(30, fields.pager_transfer_reservations);
  }
  if (status.ok()) status = builder->add_hash(31, fields.profile_envelope_root);
  if (status.ok()) {
    status = builder->add_hash(32, fields.device_observation_root);
  }
  if (status.ok()) {
    status = builder->add_hash(33, fields.capacity_plan_instance_root);
  }
  if (status.ok()) {
    status = builder->add_hash(34, fields.post_mapping_seal_root);
  }
  if (status.ok()) {
    status = builder->add_hash(35, fields.metadata_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(36, fields.mapping_owner_root);
  if (status.ok()) status = builder->add_hash(37, fields.grant_root);
  if (status.ok()) status = builder->add_hash(38, fields.prefault_layout_root);
  if (status.ok()) {
    status = builder->add_hash(39, fields.prefault_receipt_root);
  }
  if (status.ok()) status = builder->add_hash(40, fields.weight_layout_root);
  if (status.ok()) status = builder->add_hash(41, fields.weight_seal_root);
  if (status.ok()) status = builder->add_hash(42, fields.cuda_allocation_root);
  if (status.ok()) {
    // Zero is the canonical full-resident sentinel, so this fixed-width
    // optional commitment is bytes rather than the nonzero hash field type.
    status = builder->add_bytes(43, fields.pinned_allocation_root.bytes);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<DeepSeekRankMaterializationCompletionFields>
compile_deepseek_rank_materialization_completion(
    const DeepSeekRankEngineResources& resources,
    const DeepSeekRankMaterializationCompletionObservation& observation) {
  const auto* admission = resources.materialization_admission();
  const auto* prefault = resources.prefault_receipt();
  const auto* pager = resources.expert_pager();
  if (!resources.materialization_authorized() || admission == nullptr ||
      prefault == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek materialization completion requires authorized resources");
  }
  auto status = resources.verify_weight_seal();
  if (!status.ok()) return status;
  auto allocation_census =
      resources.capture_materialization_allocation_census();
  if (!allocation_census.ok()) return allocation_census.status();
  const auto& grant = admission->fields();
  const auto& arena = resources.resident_weights();
  const auto& receipt = resources.receipt();
  const bool host_spill =
      admission->residency() == RuntimeProfileResidency::kHostSpill;
  const bool pager_shape =
      host_spill
          ? pager != nullptr && !pager->poisoned() &&
                pager->staging_extent_count() ==
                    resources.staging_extent_count()
          : pager == nullptr;
  const auto expected_residency =
      host_spill ? DeepSeekRoutedExpertResidency::kHostSpill
                 : DeepSeekRoutedExpertResidency::kFullResident;
  if (!pager_shape ||
      resources.expert_residency() != expected_residency ||
      resources.rank() != admission->rank() ||
      receipt.epoch() != admission->engine_epoch() ||
      receipt.world_size() != admission->world_size() ||
      receipt.rank() != admission->rank() ||
      receipt.mapped_bytes() != prefault->mapped_interval_bytes() ||
      arena.device().type() != DeviceType::kCuda ||
      arena.device().index() != admission->device_ordinal() ||
      observation.cuda_device_ordinal != admission->device_ordinal() ||
      observation.cuda_resident_weight_allocation_bytes !=
          arena.backing_bytes() ||
      observation.cuda_resident_weight_allocation_bytes !=
          allocation_census->cuda_resident_weight_allocation_bytes ||
      observation.pinned_staging_allocation_bytes !=
          resources.pinned_staging_bytes() ||
      observation.pinned_staging_allocation_bytes !=
          allocation_census->pinned_staging_allocation_bytes ||
      observation.cuda_allocation_root !=
          allocation_census->cuda_allocation_root ||
      observation.pinned_allocation_root !=
          allocation_census->pinned_allocation_root ||
      observation.resident_selected_page_bytes !=
          prefault->selected_page_union_bytes() ||
      observation.major_fault_count != prefault->after().major_fault_count ||
      observation.completion_monotonic_ns <=
          prefault->after().monotonic_ns ||
      observation.completion_monotonic_ns >= admission->deadline_ns()) {
    return Status::FailedPrecondition(
        "DeepSeek materialization completion observation drifted");
  }

  DeepSeekRankMaterializationCompletionFields fields{
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
      grant.device_ordinal,
      grant.gpu_family,
      grant.residency,
      grant.production_eligible,
      grant.dspark_enabled,
      prefault->after().monotonic_ns,
      observation.completion_monotonic_ns,
      grant.deadline_ns,
      prefault->mapped_interval_bytes(),
      prefault->selected_page_union_bytes(),
      observation.resident_selected_page_bytes,
      prefault->after().major_fault_count,
      observation.major_fault_count,
      arena.backing_bytes(),
      arena.payload_bytes(),
      arena.generation(),
      resources.pinned_staging_bytes(),
      resources.pinned_staging_generation(),
      pager == nullptr ? 0U : pager->slot_count(),
      resources.staging_extent_count(),
      pager == nullptr ? 0U : pager->transfer_reservations(),
      grant.profile_envelope_root,
      grant.device_observation_root,
      grant.capacity_plan_instance_root,
      grant.post_mapping_seal_root,
      grant.metadata_transaction_root,
      grant.mapping_owner_root,
      admission->grant_root(),
      prefault->layout_root(),
      prefault->receipt_root(),
      receipt.weight_layout_digest(),
      receipt.weight_seal_digest(),
      observation.cuda_allocation_root,
      observation.pinned_allocation_root};
  auto root = compile_deepseek_rank_materialization_completion_root(fields);
  if (!root.ok()) return root.status();
  return fields;
}

Result<std::array<
    std::byte, kDeepSeekRankMaterializationCompletionFrameBytes>>
encode_deepseek_rank_materialization_completion(
    const DeepSeekRankMaterializationCompletionFields& fields) {
  auto root = compile_deepseek_rank_materialization_completion_root(fields);
  if (!root.ok()) return root.status();
  std::array<std::byte,
             kDeepSeekRankMaterializationCompletionFrameBytes> output{};
  std::size_t offset = 0;
  put(output, offset, kMagic);
  put(output, offset, kFrameType);
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
  put(output, offset, fields.device_ordinal);
  put(output, offset, static_cast<std::uint32_t>(fields.gpu_family));
  put(output, offset, static_cast<std::uint32_t>(fields.residency));
  put(output, offset,
      static_cast<std::uint32_t>(fields.production_eligible ? 1U : 0U));
  put(output, offset,
      static_cast<std::uint32_t>(fields.dspark_enabled ? 1U : 0U));
  put(output, offset, fields.prefault_completed_monotonic_ns);
  put(output, offset, fields.completion_monotonic_ns);
  put(output, offset, fields.deadline_ns);
  put(output, offset, fields.mapped_interval_bytes);
  put(output, offset, fields.selected_page_union_bytes);
  put(output, offset, fields.resident_selected_page_bytes);
  put(output, offset, fields.prefault_major_fault_count);
  put(output, offset, fields.completion_major_fault_count);
  put(output, offset, fields.fixed_weight_backing_bytes);
  put(output, offset, fields.fixed_weight_payload_bytes);
  put(output, offset, fields.fixed_weight_allocation_generation);
  put(output, offset, fields.pinned_staging_bytes);
  put(output, offset, fields.pinned_staging_allocation_generation);
  put(output, offset, fields.expert_slot_count);
  put(output, offset, fields.staging_extent_count);
  put(output, offset, fields.pager_transfer_reservations);
  put_digest(output, offset, fields.profile_envelope_root);
  put_digest(output, offset, fields.device_observation_root);
  put_digest(output, offset, fields.capacity_plan_instance_root);
  put_digest(output, offset, fields.post_mapping_seal_root);
  put_digest(output, offset, fields.metadata_transaction_root);
  put_digest(output, offset, fields.mapping_owner_root);
  put_digest(output, offset, fields.grant_root);
  put_digest(output, offset, fields.prefault_layout_root);
  put_digest(output, offset, fields.prefault_receipt_root);
  put_digest(output, offset, fields.weight_layout_root);
  put_digest(output, offset, fields.weight_seal_root);
  put_digest(output, offset, fields.cuda_allocation_root);
  put_digest(output, offset, fields.pinned_allocation_root);
  put_digest(output, offset, *root);
  return output;
}

Result<DeepSeekRankMaterializationCompletionFields>
decode_deepseek_rank_materialization_completion(
    std::span<const std::byte> frame) {
  if (frame.size() != kDeepSeekRankMaterializationCompletionFrameBytes) {
    return Status::InvalidArgument(
        "DeepSeek materialization completion frame size is invalid");
  }
  std::size_t offset = 0;
  if (get<std::uint32_t>(frame, offset) != kMagic ||
      get<std::uint16_t>(frame, offset) != kFrameType ||
      get<std::uint16_t>(frame, offset) != 1) {
    return Status::InvalidArgument(
        "DeepSeek materialization completion frame header is invalid");
  }
  DeepSeekRankMaterializationCompletionFields fields{};
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
  const auto device_ordinal = get<std::uint32_t>(frame, offset);
  const auto gpu_family = get<std::uint32_t>(frame, offset);
  const auto residency = get<std::uint32_t>(frame, offset);
  const auto production_eligible = get<std::uint32_t>(frame, offset);
  const auto dspark_enabled = get<std::uint32_t>(frame, offset);
  fields.prefault_completed_monotonic_ns = get<std::uint64_t>(frame, offset);
  fields.completion_monotonic_ns = get<std::uint64_t>(frame, offset);
  fields.deadline_ns = get<std::uint64_t>(frame, offset);
  fields.mapped_interval_bytes = get<std::uint64_t>(frame, offset);
  fields.selected_page_union_bytes = get<std::uint64_t>(frame, offset);
  fields.resident_selected_page_bytes = get<std::uint64_t>(frame, offset);
  fields.prefault_major_fault_count = get<std::uint64_t>(frame, offset);
  fields.completion_major_fault_count = get<std::uint64_t>(frame, offset);
  fields.fixed_weight_backing_bytes = get<std::uint64_t>(frame, offset);
  fields.fixed_weight_payload_bytes = get<std::uint64_t>(frame, offset);
  fields.fixed_weight_allocation_generation =
      get<std::uint64_t>(frame, offset);
  fields.pinned_staging_bytes = get<std::uint64_t>(frame, offset);
  fields.pinned_staging_allocation_generation =
      get<std::uint64_t>(frame, offset);
  fields.expert_slot_count = get<std::uint32_t>(frame, offset);
  fields.staging_extent_count = get<std::uint32_t>(frame, offset);
  fields.pager_transfer_reservations = get<std::uint32_t>(frame, offset);
  fields.profile_envelope_root = get_digest(frame, offset);
  fields.device_observation_root = get_digest(frame, offset);
  fields.capacity_plan_instance_root = get_digest(frame, offset);
  fields.post_mapping_seal_root = get_digest(frame, offset);
  fields.metadata_transaction_root = get_digest(frame, offset);
  fields.mapping_owner_root = get_digest(frame, offset);
  fields.grant_root = get_digest(frame, offset);
  fields.prefault_layout_root = get_digest(frame, offset);
  fields.prefault_receipt_root = get_digest(frame, offset);
  fields.weight_layout_root = get_digest(frame, offset);
  fields.weight_seal_root = get_digest(frame, offset);
  fields.cuda_allocation_root = get_digest(frame, offset);
  fields.pinned_allocation_root = get_digest(frame, offset);
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
        "DeepSeek materialization completion enum is invalid");
  }
  fields.device_ordinal = static_cast<std::int32_t>(device_ordinal);
  fields.gpu_family = static_cast<RuntimeProfileGpuFamily>(gpu_family);
  fields.residency = static_cast<RuntimeProfileResidency>(residency);
  fields.production_eligible = production_eligible == 1;
  fields.dspark_enabled = dspark_enabled == 1;
  auto root = compile_deepseek_rank_materialization_completion_root(fields);
  if (!root.ok()) return root.status();
  if (*root != encoded_root) {
    return Status::FailedPrecondition(
        "DeepSeek materialization completion root differs");
  }
  return fields;
}

Result<DeepSeekRankMaterializationCompletionSender>
DeepSeekRankMaterializationCompletionSender::Create(
    DeepSeekRankMaterializationCompletionFields completion,
    std::int32_t control_fd,
    DeepSeekRankMaterializationCompletionSenderOperations& operations) {
  if (control_fd < 0) {
    return Status::InvalidArgument(
        "DeepSeek materialization completion control descriptor is invalid");
  }
  auto frame = encode_deepseek_rank_materialization_completion(completion);
  if (!frame.ok()) return frame.status();
  return DeepSeekRankMaterializationCompletionSender(
      std::move(completion), std::move(*frame), control_fd, operations);
}

Status DeepSeekRankMaterializationCompletionSender::advance() {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek materialization completion sender is poisoned");
  }
  if (complete_) return Status::Ok();
  auto now = operations_->monotonic_now_ns();
  if (!now.ok()) {
    poisoned_ = true;
    return now.status();
  }
  if (*now >= completion_.deadline_ns) {
    poisoned_ = true;
    return Status::DeadlineExceeded(
        "DeepSeek materialization completion send deadline expired");
  }
  auto status = operations_->send_completion(control_fd_, frame_);
  if (!status.ok()) {
    if (status.code() != StatusCode::kUnavailable) poisoned_ = true;
    return status;
  }
  now = operations_->monotonic_now_ns();
  if (!now.ok()) {
    poisoned_ = true;
    return now.status();
  }
  if (*now >= completion_.deadline_ns) {
    poisoned_ = true;
    return Status::DeadlineExceeded(
        "DeepSeek materialization completion crossed its send deadline");
  }
  complete_ = true;
  return Status::Ok();
}

}  // namespace pih
