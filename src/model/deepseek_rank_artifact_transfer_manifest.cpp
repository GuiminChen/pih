#include "pih/model/deepseek_rank_artifact_transfer_manifest.h"

#include <type_traits>
#include <utility>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x50524958U;
constexpr std::uint16_t kFrameType = 5;
constexpr std::uint16_t kFrameVersion = 1;

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

Result<Sha256Digest> compile_root(
    const DeepSeekRankArtifactTransferManifestFields& fields) {
  if (fields.engine_epoch == 0 || fields.worker_generation == 0 ||
      fields.world_size < 1 || fields.world_size > 4 ||
      fields.rank >= fields.world_size ||
      fields.process_manifest_identity == 0 || fields.process_identity == 0 ||
      fields.pidfd_identity == 0 || fields.control_identity == 0 ||
      fields.challenge_identity == 0 || fields.deadline_ns == 0 ||
      fields.descriptor_count == 0 ||
      fields.descriptor_count >
          kDeepSeekRankArtifactTransferDescriptorMaximum ||
      fields.tensor_record_count == 0 ||
      fields.tensor_record_count >
          DeepSeekRuntimeRecordsManifest::kMaximumRecordCount ||
      fields.descriptor_batch_maximum !=
          kDeepSeekRankArtifactTransferDescriptorBatchMaximum ||
      fields.descriptor_batch_count !=
          (fields.descriptor_count +
           kDeepSeekRankArtifactTransferDescriptorBatchMaximum - 1U) /
              kDeepSeekRankArtifactTransferDescriptorBatchMaximum ||
      !nonzero(fields.model_startup_plan_root) ||
      !nonzero(fields.model_startup_rank_seed_root) ||
      !nonzero(fields.capacity_plan_instance_root) ||
      !nonzero(fields.artifact_admission_binding_root) ||
      !nonzero(fields.artifact_handoff_plan_root) ||
      !nonzero(fields.artifact_handoff_rank_root) ||
      !nonzero(fields.artifact_root) || !nonzero(fields.mapping_root)) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact transfer manifest is invalid");
  }
  switch (fields.immutability_mode) {
    case ArtifactImmutabilityMode::kUncalibrated:
      if (fields.source_catalog_production_eligible) {
        return Status::InvalidArgument(
            "uncalibrated DeepSeek artifact source claims production eligibility");
      }
      break;
    case ArtifactImmutabilityMode::kFsVerity:
      if (!fields.source_catalog_production_eligible) {
        return Status::InvalidArgument(
            "fs-verity DeepSeek artifact source lacks production eligibility");
      }
      break;
    case ArtifactImmutabilityMode::kDmVeritySnapshot:
      return Status::InvalidArgument(
          "DeepSeek target-generation dm-verity transfer is not implemented");
    default:
      return Status::InvalidArgument(
          "DeepSeek rank artifact immutability encoding is invalid");
  }

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-transfer-manifest:v1", 24);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, fields.engine_epoch);
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
  if (status.ok()) status = builder->add_u64(10, fields.deadline_ns);
  if (status.ok()) {
    status = builder->add_u32(
        11, static_cast<std::uint32_t>(fields.immutability_mode));
  }
  if (status.ok()) {
    status = builder->add_u32(
        12, fields.source_catalog_production_eligible ? 1U : 0U);
  }
  if (status.ok()) status = builder->add_u32(13, fields.descriptor_count);
  if (status.ok()) status = builder->add_u32(14, fields.tensor_record_count);
  if (status.ok()) {
    status = builder->add_u32(15, fields.descriptor_batch_maximum);
  }
  if (status.ok()) {
    status = builder->add_u32(16, fields.descriptor_batch_count);
  }
  if (status.ok()) {
    status = builder->add_hash(17, fields.model_startup_plan_root);
  }
  if (status.ok()) {
    status = builder->add_hash(18, fields.model_startup_rank_seed_root);
  }
  if (status.ok()) {
    status = builder->add_hash(19, fields.capacity_plan_instance_root);
  }
  if (status.ok()) {
    status = builder->add_hash(20, fields.artifact_admission_binding_root);
  }
  if (status.ok()) {
    status = builder->add_hash(21, fields.artifact_handoff_plan_root);
  }
  if (status.ok()) {
    status = builder->add_hash(22, fields.artifact_handoff_rank_root);
  }
  if (status.ok()) status = builder->add_hash(23, fields.artifact_root);
  if (status.ok()) status = builder->add_hash(24, fields.mapping_root);
  if (!status.ok()) return status;
  return builder->finalize();
}

template <class T, std::size_t N>
void put(std::array<std::byte, N>& output, std::size_t& cursor, T value) {
  using U = std::make_unsigned_t<T>;
  const auto bits = static_cast<U>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output[cursor++] =
        static_cast<std::byte>((bits >> (index * 8U)) & 0xffU);
  }
}

template <class T>
T get(std::span<const std::byte> input, std::size_t& cursor) {
  using U = std::make_unsigned_t<T>;
  U value = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    value |= static_cast<U>(std::to_integer<unsigned>(input[cursor++]))
             << (index * 8U);
  }
  return static_cast<T>(value);
}

template <std::size_t N>
void put_digest(std::array<std::byte, N>& output, std::size_t& cursor,
                const Sha256Digest& value) {
  for (const auto byte : value.bytes) output[cursor++] = byte;
}

Sha256Digest get_digest(std::span<const std::byte> input,
                        std::size_t& cursor) {
  Sha256Digest value{};
  for (auto& byte : value.bytes) byte = input[cursor++];
  return value;
}

}  // namespace

DeepSeekRankArtifactTransferManifest::DeepSeekRankArtifactTransferManifest(
    DeepSeekRankArtifactTransferManifestFields fields,
    Sha256Digest manifest_root) noexcept
    : fields_(std::move(fields)), manifest_root_(manifest_root) {}

Result<DeepSeekRankArtifactTransferManifest>
DeepSeekRankArtifactTransferManifest::Create(
    DeepSeekRankArtifactTransferManifestFields fields) {
  auto root = compile_root(fields);
  if (!root.ok()) return root.status();
  return DeepSeekRankArtifactTransferManifest(std::move(fields), *root);
}

std::array<std::byte, kDeepSeekRankArtifactTransferManifestBytes>
encode_deepseek_rank_artifact_transfer_manifest(
    const DeepSeekRankArtifactTransferManifest& manifest) {
  std::array<std::byte, kDeepSeekRankArtifactTransferManifestBytes> output{};
  std::size_t cursor = 0;
  put(output, cursor, kMagic);
  put(output, cursor, kFrameType);
  put(output, cursor, kFrameVersion);
  const auto& fields = manifest.fields();
  put(output, cursor, fields.engine_epoch);
  put(output, cursor, fields.worker_generation);
  put(output, cursor, fields.world_size);
  put(output, cursor, fields.rank);
  put(output, cursor, fields.process_manifest_identity);
  put(output, cursor, fields.process_identity);
  put(output, cursor, fields.pidfd_identity);
  put(output, cursor, fields.control_identity);
  put(output, cursor, fields.challenge_identity);
  put(output, cursor, fields.deadline_ns);
  put(output, cursor, static_cast<std::uint32_t>(fields.immutability_mode));
  put(output, cursor,
      static_cast<std::uint32_t>(
          fields.source_catalog_production_eligible ? 1U : 0U));
  put(output, cursor, fields.descriptor_count);
  put(output, cursor, fields.tensor_record_count);
  put(output, cursor, fields.descriptor_batch_maximum);
  put(output, cursor, fields.descriptor_batch_count);
  put_digest(output, cursor, fields.model_startup_plan_root);
  put_digest(output, cursor, fields.model_startup_rank_seed_root);
  put_digest(output, cursor, fields.capacity_plan_instance_root);
  put_digest(output, cursor, fields.artifact_admission_binding_root);
  put_digest(output, cursor, fields.artifact_handoff_plan_root);
  put_digest(output, cursor, fields.artifact_handoff_rank_root);
  put_digest(output, cursor, fields.artifact_root);
  put_digest(output, cursor, fields.mapping_root);
  put_digest(output, cursor, manifest.manifest_root());
  return output;
}

Result<DeepSeekRankArtifactTransferManifest>
decode_deepseek_rank_artifact_transfer_manifest(
    std::span<const std::byte> frame) {
  if (frame.size() != kDeepSeekRankArtifactTransferManifestBytes) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact transfer frame size is invalid");
  }
  std::size_t cursor = 0;
  if (get<std::uint32_t>(frame, cursor) != kMagic ||
      get<std::uint16_t>(frame, cursor) != kFrameType ||
      get<std::uint16_t>(frame, cursor) != kFrameVersion) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact transfer frame header is invalid");
  }
  DeepSeekRankArtifactTransferManifestFields fields;
  fields.engine_epoch = get<std::uint64_t>(frame, cursor);
  fields.worker_generation = get<std::uint64_t>(frame, cursor);
  fields.world_size = get<std::uint32_t>(frame, cursor);
  fields.rank = get<std::uint32_t>(frame, cursor);
  fields.process_manifest_identity = get<std::uint64_t>(frame, cursor);
  fields.process_identity = get<std::uint64_t>(frame, cursor);
  fields.pidfd_identity = get<std::uint64_t>(frame, cursor);
  fields.control_identity = get<std::uint64_t>(frame, cursor);
  fields.challenge_identity = get<std::uint64_t>(frame, cursor);
  fields.deadline_ns = get<std::uint64_t>(frame, cursor);
  const auto immutability_mode = get<std::uint32_t>(frame, cursor);
  if (immutability_mode >
      static_cast<std::uint32_t>(ArtifactImmutabilityMode::kDmVeritySnapshot)) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact immutability encoding is invalid");
  }
  fields.immutability_mode =
      static_cast<ArtifactImmutabilityMode>(immutability_mode);
  const auto production_eligible = get<std::uint32_t>(frame, cursor);
  if (production_eligible > 1U) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact source eligibility encoding is invalid");
  }
  fields.source_catalog_production_eligible = production_eligible == 1U;
  fields.descriptor_count = get<std::uint32_t>(frame, cursor);
  fields.tensor_record_count = get<std::uint32_t>(frame, cursor);
  fields.descriptor_batch_maximum = get<std::uint32_t>(frame, cursor);
  fields.descriptor_batch_count = get<std::uint32_t>(frame, cursor);
  fields.model_startup_plan_root = get_digest(frame, cursor);
  fields.model_startup_rank_seed_root = get_digest(frame, cursor);
  fields.capacity_plan_instance_root = get_digest(frame, cursor);
  fields.artifact_admission_binding_root = get_digest(frame, cursor);
  fields.artifact_handoff_plan_root = get_digest(frame, cursor);
  fields.artifact_handoff_rank_root = get_digest(frame, cursor);
  fields.artifact_root = get_digest(frame, cursor);
  fields.mapping_root = get_digest(frame, cursor);
  const auto expected_root = get_digest(frame, cursor);
  auto result = DeepSeekRankArtifactTransferManifest::Create(std::move(fields));
  if (!result.ok()) return result.status();
  if (result->manifest_root() != expected_root) {
    return Status::FailedPrecondition(
        "DeepSeek rank artifact transfer frame root differs");
  }
  return result;
}

static_assert(kDeepSeekRankArtifactTransferManifestBytes ==
              8 + (2 * 8) + (2 * 4) + (5 * 8) + 8 + (6 * 4) +
                  (9 * 32));

}  // namespace pih
