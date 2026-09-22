#include "pih/model/deepseek_rank_artifact_metadata_blob.h"

#include <algorithm>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x50524958U;
constexpr std::uint16_t kFrameType = 8;
constexpr std::uint16_t kFrameVersion = 1;
constexpr std::size_t kDigestFieldCount = 9;
constexpr std::size_t kFixedHeaderBytes =
    8 + (2 * 8) + (3 * 4) + (5 * 8) + (5 * 4) + (2 * 8) +
    kDigestFieldCount * 32;
constexpr std::size_t kTrailingRootBytes = 32;

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

bool stage_owns(const DeepSeekStagePlan& stage,
                const DeepSeekRankTensorRecord& record) noexcept {
  switch (record.role) {
    case DeepSeekTensorRole::kEmbedding:
      return stage.owns_embedding && record.logical_layer == UINT32_MAX;
    case DeepSeekTensorRole::kFinalHead:
      return stage.owns_lm_head && record.logical_layer == UINT32_MAX;
    case DeepSeekTensorRole::kMainLayer:
      return record.logical_layer >= stage.layers.first_layer &&
             record.logical_layer <= stage.layers.last_layer;
    case DeepSeekTensorRole::kDspark:
      return stage.owns_dspark && record.logical_layer <= 2U;
  }
  return false;
}

Result<Sha256Digest> validate_and_compile_root(
    const DeepSeekRankArtifactMetadataBlobFields& fields,
    const DeepSeekRankMappingPlan& mapping,
    std::span<const DeepSeekRankTensorRecord> tensor_records) {
  if (fields.engine_epoch == 0 || fields.worker_generation == 0 ||
      fields.world_size < 1 || fields.world_size > 4 ||
      fields.rank >= fields.world_size || mapping.rank != fields.rank ||
      fields.process_manifest_identity == 0 || fields.process_identity == 0 ||
      fields.pidfd_identity == 0 || fields.control_identity == 0 ||
      fields.challenge_identity == 0 || fields.descriptor_count == 0 ||
      fields.descriptor_count >
          DeepSeekRuntimeArtifactManifest::kMaximumTargetMemberCount ||
      fields.descriptor_count != mapping.shards.size() ||
      tensor_records.empty() ||
      tensor_records.size() >
          DeepSeekRuntimeRecordsManifest::kMaximumRecordCount ||
      mapping.owned_tensor_count != tensor_records.size() ||
      !nonzero(fields.transfer_manifest_root) ||
      !nonzero(fields.artifact_admission_binding_root) ||
      !nonzero(fields.descriptor_transfer_transaction_root) ||
      !nonzero(fields.artifact_handoff_rank_root) ||
      !nonzero(fields.artifact_root) || !nonzero(fields.mapping_root) ||
      !nonzero(fields.rank_mapping_root) ||
      !nonzero(fields.descriptor_handoff_root) ||
      !nonzero(fields.tensor_handoff_root)) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact metadata fields are invalid");
  }
  auto pipeline = DeepSeekPipelinePlan::Create(
      fields.world_size, fields.dspark_enabled);
  if (!pipeline.ok()) return pipeline.status();
  const auto& stage = pipeline->rank(fields.rank);
  auto rank_mapping_root =
      compile_deepseek_rank_mapping_plan_root(mapping);
  if (!rank_mapping_root.ok()) return rank_mapping_root.status();
  auto tensor_root =
      compile_deepseek_rank_tensor_handoff_root(tensor_records);
  if (!tensor_root.ok()) return tensor_root.status();
  auto handoff_root =
      compile_deepseek_rank_artifact_handoff_manifest_root_from_roots(
          fields.rank, *rank_mapping_root,
          fields.descriptor_handoff_root, *tensor_root,
          fields.descriptor_count,
          static_cast<std::uint32_t>(tensor_records.size()));
  if (!handoff_root.ok()) return handoff_root.status();
  if (*rank_mapping_root != fields.rank_mapping_root ||
      *tensor_root != fields.tensor_handoff_root ||
      *handoff_root != fields.artifact_handoff_rank_root) {
    return Status::FailedPrecondition(
        "DeepSeek rank artifact metadata roots differ");
  }

  std::map<std::string_view, std::uint64_t, std::less<>> shard_bytes;
  for (const auto& shard : mapping.shards) {
    shard_bytes.emplace(shard.shard_name, shard.file_bytes);
  }
  std::uint64_t logical_bytes = 0;
  for (const auto& record : tensor_records) {
    const auto shard = shard_bytes.find(record.shard_name);
    if (!stage_owns(stage, record) || shard == shard_bytes.end() ||
        record.artifact_root != fields.artifact_root ||
        record.file_begin >= record.file_end ||
        record.file_end > shard->second) {
      return Status::FailedPrecondition(
          "DeepSeek rank artifact metadata tensor ownership differs");
    }
    auto total = checked_add_u64(logical_bytes, record.tensor_bytes);
    if (!total.ok()) return total.status();
    logical_bytes = *total;
    const auto covered = std::ranges::any_of(
        mapping.intervals, [&](const DeepSeekMappedInterval& interval) {
          return interval.shard_name == record.shard_name &&
                 interval.file_begin <= record.file_begin &&
                 interval.file_end >= record.file_end;
        });
    if (!covered) {
      return Status::FailedPrecondition(
          "DeepSeek rank artifact metadata tensor is not mapped");
    }
  }
  if (logical_bytes != mapping.logical_tensor_bytes) {
    return Status::FailedPrecondition(
        "DeepSeek rank artifact metadata logical bytes differ");
  }

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-metadata-blob:v1", 26);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, fields.engine_epoch);
  if (status.ok()) status = builder->add_u64(2, fields.worker_generation);
  if (status.ok()) status = builder->add_u32(3, fields.world_size);
  if (status.ok()) status = builder->add_u32(4, fields.rank);
  if (status.ok()) status = builder->add_u32(5, fields.dspark_enabled ? 1U : 0U);
  if (status.ok()) {
    status = builder->add_u64(6, fields.process_manifest_identity);
  }
  if (status.ok()) status = builder->add_u64(7, fields.process_identity);
  if (status.ok()) status = builder->add_u64(8, fields.pidfd_identity);
  if (status.ok()) status = builder->add_u64(9, fields.control_identity);
  if (status.ok()) status = builder->add_u64(10, fields.challenge_identity);
  if (status.ok()) status = builder->add_u32(11, fields.descriptor_count);
  if (status.ok()) status = builder->add_u32(12, mapping.owned_tensor_count);
  if (status.ok()) status = builder->add_u64(13, mapping.logical_tensor_bytes);
  if (status.ok()) status = builder->add_u64(14, mapping.mapped_interval_bytes);
  if (status.ok()) {
    status = builder->add_u32(15, static_cast<std::uint32_t>(mapping.shards.size()));
  }
  if (status.ok()) {
    status = builder->add_u32(16, static_cast<std::uint32_t>(mapping.intervals.size()));
  }
  if (status.ok()) {
    status = builder->add_u32(17, static_cast<std::uint32_t>(tensor_records.size()));
  }
  if (status.ok()) status = builder->add_hash(18, fields.transfer_manifest_root);
  if (status.ok()) {
    status = builder->add_hash(19, fields.artifact_admission_binding_root);
  }
  if (status.ok()) {
    status = builder->add_hash(20, fields.descriptor_transfer_transaction_root);
  }
  if (status.ok()) status = builder->add_hash(21, fields.artifact_handoff_rank_root);
  if (status.ok()) status = builder->add_hash(22, fields.artifact_root);
  if (status.ok()) status = builder->add_hash(23, fields.mapping_root);
  if (status.ok()) status = builder->add_hash(24, fields.rank_mapping_root);
  if (status.ok()) status = builder->add_hash(25, fields.descriptor_handoff_root);
  if (status.ok()) status = builder->add_hash(26, fields.tensor_handoff_root);
  if (!status.ok()) return status;
  return builder->finalize();
}

template <class T>
void put(std::vector<std::byte>& output, T value) {
  using U = std::make_unsigned_t<T>;
  const auto bits = static_cast<U>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output.push_back(
        static_cast<std::byte>((bits >> (index * 8U)) & 0xffU));
  }
}

void put_digest(std::vector<std::byte>& output,
                const Sha256Digest& value) {
  output.insert(output.end(), value.bytes.begin(), value.bytes.end());
}

void put_string(std::vector<std::byte>& output, std::string_view value) {
  put(output, static_cast<std::uint16_t>(value.size()));
  for (const unsigned char byte : value) {
    output.push_back(static_cast<std::byte>(byte));
  }
}

template <class T>
bool get(std::span<const std::byte> input, std::size_t& cursor, T* value) {
  if (value == nullptr || cursor > input.size() ||
      input.size() - cursor < sizeof(T)) {
    return false;
  }
  using U = std::make_unsigned_t<T>;
  U bits = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    bits |= static_cast<U>(std::to_integer<unsigned>(input[cursor++]))
            << (index * 8U);
  }
  *value = static_cast<T>(bits);
  return true;
}

bool get_digest(std::span<const std::byte> input, std::size_t& cursor,
                Sha256Digest* value) {
  if (value == nullptr || cursor > input.size() ||
      input.size() - cursor < value->bytes.size()) {
    return false;
  }
  for (auto& byte : value->bytes) byte = input[cursor++];
  return true;
}

bool get_string(std::span<const std::byte> input, std::size_t& cursor,
                std::size_t maximum_bytes, std::string* value) {
  std::uint16_t bytes = 0;
  if (value == nullptr || !get(input, cursor, &bytes) || bytes == 0 ||
      bytes > maximum_bytes || cursor > input.size() ||
      input.size() - cursor < bytes) {
    return false;
  }
  value->clear();
  value->reserve(bytes);
  for (std::uint16_t index = 0; index < bytes; ++index) {
    const auto byte = std::to_integer<unsigned>(input[cursor++]);
    if (byte == 0 || byte > 0x7fU) return false;
    value->push_back(static_cast<char>(byte));
  }
  return true;
}

}  // namespace

DeepSeekRankArtifactMetadataBlob::DeepSeekRankArtifactMetadataBlob(
    DeepSeekRankArtifactMetadataBlobFields fields,
    DeepSeekRankMappingPlan mapping,
    std::vector<DeepSeekRankTensorRecord> tensor_records,
    Sha256Digest metadata_root) noexcept
    : fields_(std::move(fields)), mapping_(std::move(mapping)),
      tensor_records_(std::move(tensor_records)),
      metadata_root_(metadata_root) {}

Result<DeepSeekRankArtifactMetadataBlob>
DeepSeekRankArtifactMetadataBlob::Create(
    DeepSeekRankArtifactMetadataBlobFields fields,
    DeepSeekRankMappingPlan mapping,
    std::vector<DeepSeekRankTensorRecord> tensor_records) {
  auto root = validate_and_compile_root(fields, mapping, tensor_records);
  if (!root.ok()) return root.status();
  return DeepSeekRankArtifactMetadataBlob(
      std::move(fields), std::move(mapping), std::move(tensor_records),
      *root);
}

Result<std::vector<std::byte>> encode_deepseek_rank_artifact_metadata_blob(
    const DeepSeekRankArtifactMetadataBlob& blob) {
  auto root = validate_and_compile_root(
      blob.fields(), blob.mapping(), blob.tensor_records());
  if (!root.ok()) return root.status();
  if (*root != blob.metadata_root()) {
    return Status::FailedPrecondition(
        "DeepSeek rank artifact metadata object root differs");
  }
  std::vector<std::byte> output;
  output.reserve(kFixedHeaderBytes + kTrailingRootBytes);
  put(output, kMagic);
  put(output, kFrameType);
  put(output, kFrameVersion);
  const auto& fields = blob.fields();
  const auto& mapping = blob.mapping();
  put(output, fields.engine_epoch);
  put(output, fields.worker_generation);
  put(output, fields.world_size);
  put(output, fields.rank);
  put(output, static_cast<std::uint32_t>(fields.dspark_enabled ? 1U : 0U));
  put(output, fields.process_manifest_identity);
  put(output, fields.process_identity);
  put(output, fields.pidfd_identity);
  put(output, fields.control_identity);
  put(output, fields.challenge_identity);
  put(output, fields.descriptor_count);
  put(output, mapping.owned_tensor_count);
  put(output, mapping.logical_tensor_bytes);
  put(output, mapping.mapped_interval_bytes);
  put(output, static_cast<std::uint32_t>(mapping.shards.size()));
  put(output, static_cast<std::uint32_t>(mapping.intervals.size()));
  put(output, static_cast<std::uint32_t>(blob.tensor_records().size()));
  put_digest(output, fields.transfer_manifest_root);
  put_digest(output, fields.artifact_admission_binding_root);
  put_digest(output, fields.descriptor_transfer_transaction_root);
  put_digest(output, fields.artifact_handoff_rank_root);
  put_digest(output, fields.artifact_root);
  put_digest(output, fields.mapping_root);
  put_digest(output, fields.rank_mapping_root);
  put_digest(output, fields.descriptor_handoff_root);
  put_digest(output, fields.tensor_handoff_root);
  for (const auto& shard : mapping.shards) {
    put_string(output, shard.shard_name);
    put(output, shard.file_bytes);
  }
  for (const auto& interval : mapping.intervals) {
    put_string(output, interval.shard_name);
    put(output, interval.file_begin);
    put(output, interval.file_end);
  }
  for (const auto& record : blob.tensor_records()) {
    put_string(output, record.tensor_name);
    put_string(output, record.shard_name);
    put(output, static_cast<std::uint32_t>(record.role));
    put(output, static_cast<std::uint32_t>(record.dtype));
    put(output, static_cast<std::uint16_t>(record.shape.size()));
    for (const auto dimension : record.shape) put(output, dimension);
    put(output, record.file_begin);
    put(output, record.file_end);
    put(output, record.logical_layer);
    put(output, record.tensor_bytes);
    put(output, static_cast<std::uint32_t>(record.storage_semantics));
    put_digest(output, record.artifact_root);
    put_digest(output, record.layout_root);
    put_digest(output, record.disposition_root);
    put_digest(output, record.target_logical_root);
    put_digest(output, record.disposition_record_root);
    put_digest(output, record.layout_record_root);
    put_digest(output, record.runtime_record_root);
    if (output.size() >
        kDeepSeekRankArtifactMetadataBlobMaximumBytes -
            kTrailingRootBytes) {
      return Status::ResourceExhausted(
          "DeepSeek rank artifact metadata blob exceeds budget");
    }
  }
  put_digest(output, blob.metadata_root());
  if (output.size() < kFixedHeaderBytes + kTrailingRootBytes ||
      output.size() > kDeepSeekRankArtifactMetadataBlobMaximumBytes) {
    return Status::ResourceExhausted(
        "DeepSeek rank artifact metadata blob exceeds budget");
  }
  return output;
}

Result<DeepSeekRankArtifactMetadataBlob>
decode_deepseek_rank_artifact_metadata_blob(
    std::span<const std::byte> frame) {
  if (frame.size() < kFixedHeaderBytes + kTrailingRootBytes ||
      frame.size() > kDeepSeekRankArtifactMetadataBlobMaximumBytes) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact metadata blob size is invalid");
  }
  std::size_t cursor = 0;
  std::uint32_t magic = 0;
  std::uint16_t type = 0;
  std::uint16_t version = 0;
  std::uint32_t dspark_enabled = 0;
  std::uint32_t owned_tensor_count = 0;
  std::uint64_t logical_tensor_bytes = 0;
  std::uint64_t mapped_interval_bytes = 0;
  std::uint32_t shard_count = 0;
  std::uint32_t interval_count = 0;
  std::uint32_t tensor_count = 0;
  DeepSeekRankArtifactMetadataBlobFields fields;
  if (!get(frame, cursor, &magic) || !get(frame, cursor, &type) ||
      !get(frame, cursor, &version) || magic != kMagic ||
      type != kFrameType || version != kFrameVersion ||
      !get(frame, cursor, &fields.engine_epoch) ||
      !get(frame, cursor, &fields.worker_generation) ||
      !get(frame, cursor, &fields.world_size) ||
      !get(frame, cursor, &fields.rank) ||
      !get(frame, cursor, &dspark_enabled) || dspark_enabled > 1U ||
      !get(frame, cursor, &fields.process_manifest_identity) ||
      !get(frame, cursor, &fields.process_identity) ||
      !get(frame, cursor, &fields.pidfd_identity) ||
      !get(frame, cursor, &fields.control_identity) ||
      !get(frame, cursor, &fields.challenge_identity) ||
      !get(frame, cursor, &fields.descriptor_count) ||
      !get(frame, cursor, &owned_tensor_count) ||
      !get(frame, cursor, &logical_tensor_bytes) ||
      !get(frame, cursor, &mapped_interval_bytes) ||
      !get(frame, cursor, &shard_count) ||
      !get(frame, cursor, &interval_count) ||
      !get(frame, cursor, &tensor_count) ||
      !get_digest(frame, cursor, &fields.transfer_manifest_root) ||
      !get_digest(frame, cursor, &fields.artifact_admission_binding_root) ||
      !get_digest(frame, cursor,
                  &fields.descriptor_transfer_transaction_root) ||
      !get_digest(frame, cursor, &fields.artifact_handoff_rank_root) ||
      !get_digest(frame, cursor, &fields.artifact_root) ||
      !get_digest(frame, cursor, &fields.mapping_root) ||
      !get_digest(frame, cursor, &fields.rank_mapping_root) ||
      !get_digest(frame, cursor, &fields.descriptor_handoff_root) ||
      !get_digest(frame, cursor, &fields.tensor_handoff_root) ||
      fields.world_size < 1 || fields.world_size > 4 ||
      fields.rank >= fields.world_size ||
      fields.descriptor_count == 0 ||
      fields.descriptor_count >
          DeepSeekRuntimeArtifactManifest::kMaximumTargetMemberCount ||
      shard_count != fields.descriptor_count ||
      shard_count == 0 ||
      interval_count == 0 ||
      interval_count >
          DeepSeekRuntimeRecordsManifest::kMaximumRecordCount ||
      tensor_count == 0 ||
      tensor_count > DeepSeekRuntimeRecordsManifest::kMaximumRecordCount ||
      owned_tensor_count != tensor_count) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact metadata blob header is invalid");
  }
  fields.dspark_enabled = dspark_enabled == 1U;
  constexpr std::uint64_t kMinimumShardBytes = 11;
  constexpr std::uint64_t kMinimumIntervalBytes = 19;
  constexpr std::uint64_t kMinimumTensorBytes = 280;
  const auto minimum_records =
      static_cast<std::uint64_t>(shard_count) * kMinimumShardBytes +
      static_cast<std::uint64_t>(interval_count) * kMinimumIntervalBytes +
      static_cast<std::uint64_t>(tensor_count) * kMinimumTensorBytes;
  if (cursor > frame.size() ||
      minimum_records + kTrailingRootBytes > frame.size() - cursor) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact metadata record counts exceed frame");
  }

  DeepSeekRankMappingPlan mapping;
  mapping.rank = fields.rank;
  mapping.owned_tensor_count = owned_tensor_count;
  mapping.logical_tensor_bytes = logical_tensor_bytes;
  mapping.mapped_interval_bytes = mapped_interval_bytes;
  mapping.shards.reserve(shard_count);
  for (std::uint32_t index = 0; index < shard_count; ++index) {
    DeepSeekRankShardPlan shard;
    if (!get_string(frame, cursor, 255, &shard.shard_name) ||
        !get(frame, cursor, &shard.file_bytes)) {
      return Status::InvalidArgument(
          "DeepSeek rank artifact metadata shard is truncated");
    }
    mapping.shards.push_back(std::move(shard));
  }
  mapping.intervals.reserve(interval_count);
  for (std::uint32_t index = 0; index < interval_count; ++index) {
    DeepSeekMappedInterval interval;
    if (!get_string(frame, cursor, 255, &interval.shard_name) ||
        !get(frame, cursor, &interval.file_begin) ||
        !get(frame, cursor, &interval.file_end)) {
      return Status::InvalidArgument(
          "DeepSeek rank artifact metadata interval is truncated");
    }
    mapping.intervals.push_back(std::move(interval));
  }

  std::vector<DeepSeekRankTensorRecord> tensors;
  tensors.reserve(tensor_count);
  for (std::uint32_t index = 0; index < tensor_count; ++index) {
    DeepSeekRankTensorRecord record;
    std::uint32_t role = 0;
    std::uint32_t dtype = 0;
    std::uint16_t shape_count = 0;
    std::uint32_t storage = 0;
    if (!get_string(frame, cursor,
                    DeepSeekRuntimeRecordsManifest::kMaximumTensorNameBytes,
                    &record.tensor_name) ||
        !get_string(frame, cursor, 255, &record.shard_name) ||
        !get(frame, cursor, &role) || !get(frame, cursor, &dtype) ||
        !get(frame, cursor, &shape_count) || shape_count == 0 ||
        shape_count > 16) {
      return Status::InvalidArgument(
          "DeepSeek rank artifact metadata tensor header is invalid");
    }
    record.role = static_cast<DeepSeekTensorRole>(role);
    record.dtype = static_cast<DType>(dtype);
    if (static_cast<std::uint32_t>(record.role) != role ||
        static_cast<std::uint32_t>(record.dtype) != dtype) {
      return Status::InvalidArgument(
          "DeepSeek rank artifact metadata tensor enum is non-canonical");
    }
    record.shape.reserve(shape_count);
    for (std::uint16_t dimension = 0; dimension < shape_count; ++dimension) {
      std::uint64_t value = 0;
      if (!get(frame, cursor, &value)) {
        return Status::InvalidArgument(
            "DeepSeek rank artifact metadata tensor shape is truncated");
      }
      record.shape.push_back(value);
    }
    if (!get(frame, cursor, &record.file_begin) ||
        !get(frame, cursor, &record.file_end) ||
        !get(frame, cursor, &record.logical_layer) ||
        !get(frame, cursor, &record.tensor_bytes) ||
        !get(frame, cursor, &storage) ||
        !get_digest(frame, cursor, &record.artifact_root) ||
        !get_digest(frame, cursor, &record.layout_root) ||
        !get_digest(frame, cursor, &record.disposition_root) ||
        !get_digest(frame, cursor, &record.target_logical_root) ||
        !get_digest(frame, cursor, &record.disposition_record_root) ||
        !get_digest(frame, cursor, &record.layout_record_root) ||
        !get_digest(frame, cursor, &record.runtime_record_root)) {
      return Status::InvalidArgument(
          "DeepSeek rank artifact metadata tensor is truncated");
    }
    record.storage_semantics =
        static_cast<DeepSeekStorageSemantics>(storage);
    if (static_cast<std::uint32_t>(record.storage_semantics) != storage) {
      return Status::InvalidArgument(
          "DeepSeek rank artifact metadata storage enum is non-canonical");
    }
    tensors.push_back(std::move(record));
  }
  Sha256Digest encoded_root;
  if (!get_digest(frame, cursor, &encoded_root) || cursor != frame.size()) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact metadata blob has trailing bytes");
  }
  auto blob = DeepSeekRankArtifactMetadataBlob::Create(
      std::move(fields), std::move(mapping), std::move(tensors));
  if (!blob.ok()) return blob.status();
  if (blob->metadata_root() != encoded_root) {
    return Status::FailedPrecondition(
        "DeepSeek rank artifact metadata blob root differs");
  }
  return blob;
}

}  // namespace pih
