#include "pih/model/deepseek_rank_artifact_handoff_plan.h"

#include "pih/model/deepseek_controller_artifact_catalog.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

namespace pih {
namespace {

struct MappingIdentity final {
  Sha256Digest root{};
  std::vector<Sha256Digest> rank_roots;
};

bool nonzero(const Sha256Digest& value) noexcept {
  return value != Sha256Digest{};
}

Result<Sha256Digest> compile_root_set(
    std::string_view domain, std::span<const Sha256Digest> roots) {
  if (roots.size() > 65'436) {
    return Status::InvalidArgument(
        "DeepSeek handoff root set geometry is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      domain, static_cast<std::uint32_t>(roots.size() + 1U));
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(
      1, static_cast<std::uint32_t>(roots.size()));
  for (std::size_t index = 0; status.ok() && index < roots.size(); ++index) {
    status = builder->add_hash(
        static_cast<std::uint16_t>(index + 100U), roots[index]);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> compile_chunked_root_set(
    std::string_view chunk_domain, std::string_view set_domain,
    std::span<const Sha256Digest> roots) {
  constexpr std::size_t kRecordsPerChunk = 4096;
  std::vector<Sha256Digest> chunks;
  chunks.reserve((roots.size() + kRecordsPerChunk - 1U) /
                 kRecordsPerChunk);
  for (std::size_t begin = 0; begin < roots.size();
       begin += kRecordsPerChunk) {
    const auto count = std::min(kRecordsPerChunk, roots.size() - begin);
    auto chunk = compile_root_set(chunk_domain, roots.subspan(begin, count));
    if (!chunk.ok()) return chunk.status();
    chunks.push_back(*chunk);
  }
  return compile_root_set(set_domain, chunks);
}

bool valid_ascii(std::string_view value) noexcept {
  if (value.empty()) return false;
  for (const unsigned char byte : value) {
    if (byte == 0 || byte > 0x7f) return false;
  }
  return true;
}

bool valid_role_layer(DeepSeekTensorRole role,
                      std::uint32_t logical_layer) noexcept {
  switch (role) {
    case DeepSeekTensorRole::kEmbedding:
    case DeepSeekTensorRole::kFinalHead:
      return logical_layer == UINT32_MAX;
    case DeepSeekTensorRole::kMainLayer:
      return logical_layer <= 42U;
    case DeepSeekTensorRole::kDspark:
      return logical_layer <= 2U;
  }
  return false;
}

bool valid_storage(DType dtype,
                   DeepSeekStorageSemantics semantics) noexcept {
  switch (semantics) {
    case DeepSeekStorageSemantics::kDirectF32LittleEndianBits:
      return dtype == DType::kFloat32;
    case DeepSeekStorageSemantics::kDirectF16LittleEndianBits:
      return dtype == DType::kFloat16;
    case DeepSeekStorageSemantics::kDirectBf16LittleEndianBits:
      return dtype == DType::kBFloat16;
    case DeepSeekStorageSemantics::kDirectMxfp4E2m1PackedBits:
      return dtype == DType::kInt8;
    case DeepSeekStorageSemantics::kDirectU8Bits:
      return dtype == DType::kUInt8;
    case DeepSeekStorageSemantics::kDirectFp8E4m3Bits:
      return dtype == DType::kFloat8E4M3;
    case DeepSeekStorageSemantics::kDirectUe8m0ScaleBits:
      return dtype == DType::kFloat8E8M0;
    case DeepSeekStorageSemantics::kDirectI32LittleEndianBits:
      return dtype == DType::kInt32;
    case DeepSeekStorageSemantics::kDirectI64LittleEndianBits:
      return dtype == DType::kInt64;
    case DeepSeekStorageSemantics::kDirectBoolBits:
      return dtype == DType::kBool;
  }
  return false;
}

Result<Sha256Digest> compile_rank_mapping_root_impl(
    const DeepSeekRankMappingPlan& plan, std::uint32_t expected_rank) {
  if (expected_rank > 3 || plan.rank != expected_rank ||
      plan.owned_tensor_count >
          DeepSeekRuntimeRecordsManifest::kMaximumRecordCount ||
      plan.shards.size() >
          DeepSeekRuntimeArtifactManifest::kMaximumTargetMemberCount ||
      plan.intervals.size() >
          DeepSeekRuntimeRecordsManifest::kMaximumRecordCount ||
      (plan.owned_tensor_count == 0) !=
          (plan.logical_tensor_bytes == 0) ||
      (plan.shards.empty() != plan.intervals.empty()) ||
      (plan.intervals.empty() != (plan.mapped_interval_bytes == 0))) {
    return Status::InvalidArgument(
        "DeepSeek rank mapping shape is not canonical");
  }

  std::map<std::string_view, std::uint64_t, std::less<>> shard_bytes;
  auto shard_set = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-mapping-shard-set:v1",
      static_cast<std::uint32_t>(plan.shards.size() + 1U));
  if (!shard_set.ok()) return shard_set.status();
  auto status = shard_set->add_u32(
      1, static_cast<std::uint32_t>(plan.shards.size()));
  std::string_view previous_shard;
  for (std::size_t index = 0;
       status.ok() && index < plan.shards.size(); ++index) {
    const auto& shard = plan.shards[index];
    if (!valid_ascii(shard.shard_name) || shard.file_bytes == 0 ||
        (!previous_shard.empty() && shard.shard_name <= previous_shard) ||
        !shard_bytes.emplace(shard.shard_name, shard.file_bytes).second) {
      return Status::InvalidArgument(
          "DeepSeek rank mapping shard set is not canonical");
    }
    previous_shard = shard.shard_name;
    auto record = CanonicalHashBuilder::Create(
        "pih:deepseek-rank-mapping-shard:v1", 3);
    if (!record.ok()) return record.status();
    status = record->add_u32(1, static_cast<std::uint32_t>(index));
    if (status.ok()) status = record->add_ascii_utf8(2, shard.shard_name);
    if (status.ok()) status = record->add_u64(3, shard.file_bytes);
    if (!status.ok()) return status;
    auto root = record->finalize();
    if (!root.ok()) return root.status();
    status = shard_set->add_hash(
        static_cast<std::uint16_t>(index + 2U), *root);
  }
  if (!status.ok()) return status;
  auto shard_root = shard_set->finalize();
  if (!shard_root.ok()) return shard_root.status();

  std::vector<Sha256Digest> interval_roots;
  interval_roots.reserve(plan.intervals.size());
  std::uint64_t mapped_bytes = 0;
  const DeepSeekMappedInterval* previous = nullptr;
  std::set<std::string_view, std::less<>> referenced_shards;
  for (std::size_t index = 0; index < plan.intervals.size(); ++index) {
    const auto& interval = plan.intervals[index];
    const auto shard = shard_bytes.find(interval.shard_name);
    if (shard == shard_bytes.end() ||
        interval.file_begin >= interval.file_end ||
        interval.file_end > shard->second ||
        (previous != nullptr &&
         (interval.shard_name < previous->shard_name ||
          (interval.shard_name == previous->shard_name &&
           interval.file_begin <= previous->file_end)))) {
      return Status::InvalidArgument(
          "DeepSeek rank mapping interval set is not canonical");
    }
    auto total = checked_add_u64(
        mapped_bytes, interval.file_end - interval.file_begin);
    if (!total.ok()) return total.status();
    mapped_bytes = *total;
    referenced_shards.insert(interval.shard_name);
    previous = &interval;
    auto record = CanonicalHashBuilder::Create(
        "pih:deepseek-rank-mapping-interval:v1", 4);
    if (!record.ok()) return record.status();
    status = record->add_u32(1, static_cast<std::uint32_t>(index));
    if (status.ok()) status = record->add_ascii_utf8(2, interval.shard_name);
    if (status.ok()) status = record->add_u64(3, interval.file_begin);
    if (status.ok()) status = record->add_u64(4, interval.file_end);
    if (!status.ok()) return status;
    auto root = record->finalize();
    if (!root.ok()) return root.status();
    interval_roots.push_back(*root);
  }
  if (mapped_bytes != plan.mapped_interval_bytes ||
      referenced_shards.size() != shard_bytes.size()) {
    return Status::InvalidArgument(
        "DeepSeek rank mapping byte or shard ledger drifted");
  }
  auto interval_root = compile_chunked_root_set(
      "pih:deepseek-rank-mapping-interval-chunk:v1",
      "pih:deepseek-rank-mapping-interval-set:v1", interval_roots);
  if (!interval_root.ok()) return interval_root.status();

  auto rank_builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-mapping-plan:v1", 7);
  if (!rank_builder.ok()) return rank_builder.status();
  status = rank_builder->add_u32(1, expected_rank);
  if (status.ok()) status = rank_builder->add_u32(2, plan.owned_tensor_count);
  if (status.ok()) {
    status = rank_builder->add_u64(3, plan.logical_tensor_bytes);
  }
  if (status.ok()) {
    status = rank_builder->add_u64(4, plan.mapped_interval_bytes);
  }
  if (status.ok()) status = rank_builder->add_hash(5, *shard_root);
  if (status.ok()) status = rank_builder->add_hash(6, *interval_root);
  if (status.ok()) {
    status = rank_builder->add_u32(
        7, static_cast<std::uint32_t>(plan.shards.size()));
  }
  if (!status.ok()) return status;
  return rank_builder->finalize();
}

Result<MappingIdentity> compile_mapping_identity(
    const DeepSeekStageMappingPlan& mapping) {
  const auto world_size = mapping.world_size();
  if (world_size < 1 || world_size > 4) {
    return Status::InvalidArgument(
        "DeepSeek stage mapping world size is invalid");
  }

  std::vector<Sha256Digest> rank_roots;
  rank_roots.reserve(world_size);
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    auto rank_root = compile_rank_mapping_root_impl(mapping.rank(rank), rank);
    if (!rank_root.ok()) return rank_root.status();
    rank_roots.push_back(*rank_root);
  }

  auto collection = CanonicalHashBuilder::Create(
      "pih:deepseek-stage-mapping-plan:v1", world_size + 2U);
  if (!collection.ok()) return collection.status();
  auto status = collection->add_u32(1, world_size);
  if (status.ok()) {
    status = collection->add_u32(2, mapping.excluded_tensor_count());
  }
  for (std::uint32_t rank = 0; status.ok() && rank < world_size; ++rank) {
    status = collection->add_hash(
        static_cast<std::uint16_t>(rank + 3U), rank_roots[rank]);
  }
  if (!status.ok()) return status;
  auto root = collection->finalize();
  if (!root.ok()) return root.status();
  return MappingIdentity{*root, std::move(rank_roots)};
}

Result<Sha256Digest> compile_tensor_set_impl(
    std::span<const DeepSeekRankTensorRecord> records) {
  if (records.empty() ||
      records.size() > DeepSeekRuntimeRecordsManifest::kMaximumRecordCount) {
    return Status::InvalidArgument(
        "DeepSeek rank tensor handoff set geometry is invalid");
  }
  std::vector<Sha256Digest> record_roots;
  record_roots.reserve(records.size());
  std::set<std::string_view, std::less<>> names;
  std::set<std::array<std::byte, 32>> runtime_roots;
  std::string_view previous_name;
  const auto artifact_root = records.front().artifact_root;
  const auto layout_root = records.front().layout_root;
  const auto disposition_root = records.front().disposition_root;
  for (std::size_t index = 0; index < records.size(); ++index) {
    const auto& record = records[index];
    if (!valid_ascii(record.tensor_name) || !valid_ascii(record.shard_name) ||
        record.tensor_name.size() >
            DeepSeekRuntimeRecordsManifest::kMaximumTensorNameBytes ||
        record.shard_name.size() > 255U ||
        (!previous_name.empty() && record.tensor_name <= previous_name) ||
        !valid_role_layer(record.role, record.logical_layer) ||
        !valid_storage(record.dtype, record.storage_semantics) ||
        record.shape.empty() || record.shape.size() > 16 ||
        record.file_begin >= record.file_end ||
        record.file_end - record.file_begin != record.tensor_bytes ||
        !names.insert(record.tensor_name).second ||
        !runtime_roots.insert(record.runtime_record_root.bytes).second ||
        !nonzero(record.artifact_root) || !nonzero(record.layout_root) ||
        !nonzero(record.disposition_root) ||
        !nonzero(record.target_logical_root) ||
        !nonzero(record.disposition_record_root) ||
        !nonzero(record.layout_record_root) ||
        !nonzero(record.runtime_record_root) ||
        record.artifact_root != artifact_root ||
        record.layout_root != layout_root ||
        record.disposition_root != disposition_root) {
      return Status::InvalidArgument(
          "DeepSeek rank tensor handoff record is invalid");
    }
    previous_name = record.tensor_name;
    auto shape = CanonicalHashBuilder::Create(
        "pih:deepseek-rank-handoff-tensor-shape:v1",
        static_cast<std::uint32_t>(record.shape.size() + 1U));
    if (!shape.ok()) return shape.status();
    auto status = shape->add_u32(
        1, static_cast<std::uint32_t>(record.shape.size()));
    for (std::size_t dimension = 0;
         status.ok() && dimension < record.shape.size(); ++dimension) {
      if (record.shape[dimension] == 0) {
        return Status::InvalidArgument(
            "DeepSeek rank tensor handoff shape is invalid");
      }
      status = shape->add_u64(
          static_cast<std::uint16_t>(dimension + 2U),
          record.shape[dimension]);
    }
    if (!status.ok()) return status;
    auto shape_root = shape->finalize();
    if (!shape_root.ok()) return shape_root.status();

    auto item = CanonicalHashBuilder::Create(
        "pih:deepseek-rank-handoff-tensor:v1", 18);
    if (!item.ok()) return item.status();
    status = item->add_u32(1, static_cast<std::uint32_t>(index));
    if (status.ok()) status = item->add_ascii_utf8(2, record.tensor_name);
    if (status.ok()) status = item->add_ascii_utf8(3, record.shard_name);
    if (status.ok()) {
      status = item->add_u32(4, static_cast<std::uint32_t>(record.role));
    }
    if (status.ok()) {
      status = item->add_u32(5, static_cast<std::uint32_t>(record.dtype));
    }
    if (status.ok()) status = item->add_hash(6, *shape_root);
    if (status.ok()) status = item->add_u64(7, record.file_begin);
    if (status.ok()) status = item->add_u64(8, record.file_end);
    if (status.ok()) status = item->add_u32(9, record.logical_layer);
    if (status.ok()) status = item->add_u64(10, record.tensor_bytes);
    if (status.ok()) {
      status = item->add_u32(
          11, static_cast<std::uint32_t>(record.storage_semantics));
    }
    if (status.ok()) status = item->add_hash(12, record.artifact_root);
    if (status.ok()) status = item->add_hash(13, record.layout_root);
    if (status.ok()) status = item->add_hash(14, record.disposition_root);
    if (status.ok()) status = item->add_hash(15, record.target_logical_root);
    if (status.ok()) {
      status = item->add_hash(16, record.disposition_record_root);
    }
    if (status.ok()) status = item->add_hash(17, record.layout_record_root);
    if (status.ok()) status = item->add_hash(18, record.runtime_record_root);
    if (!status.ok()) return status;
    auto item_root = item->finalize();
    if (!item_root.ok()) return item_root.status();
    record_roots.push_back(*item_root);
  }

  return compile_chunked_root_set(
      "pih:deepseek-rank-handoff-tensor-chunk:v1",
      "pih:deepseek-rank-handoff-tensor-set:v1", record_roots);
}

Result<Sha256Digest> compile_descriptor_expectation_set(
    std::span<const DeepSeekRankArtifactDescriptorExpectation>
        expectations) {
  if (expectations.empty() ||
      expectations.size() >
          DeepSeekRuntimeArtifactManifest::kMaximumTargetMemberCount) {
    return Status::InvalidArgument(
        "DeepSeek rank handoff descriptor count is invalid");
  }
  auto collection = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-handoff-descriptor-set:v1",
      static_cast<std::uint32_t>(expectations.size() + 1U));
  if (!collection.ok()) return collection.status();
  auto status = collection->add_u32(
      1, static_cast<std::uint32_t>(expectations.size()));
  std::set<std::string_view, std::less<>> names;
  for (std::size_t index = 0;
       status.ok() && index < expectations.size(); ++index) {
    const auto& expectation = expectations[index];
    const auto& identity = expectation.identity;
    const auto digest_is_nonzero = nonzero(expectation.enforced_digest);
    if (expectation.ordinal != index ||
        !valid_ascii(expectation.shard_name) ||
        expectation.shard_name.size() > 255U || identity.file_bytes == 0 ||
        identity.filesystem_identity == 0 || identity.file_identity == 0 ||
        identity.data_mtime_seconds < 0 ||
        identity.data_mtime_nanoseconds >= 1'000'000'000U ||
        (expectation.immutability_mode == ArtifactImmutabilityMode::kFsVerity &&
         !digest_is_nonzero) ||
        (expectation.immutability_mode != ArtifactImmutabilityMode::kFsVerity &&
         digest_is_nonzero) ||
        !names.insert(expectation.shard_name).second) {
      return Status::InvalidArgument(
          "DeepSeek rank handoff descriptor identity is invalid");
    }
    auto record = CanonicalHashBuilder::Create(
        "pih:deepseek-rank-handoff-descriptor:v1",
        digest_is_nonzero ? 10U : 9U);
    if (!record.ok()) return record.status();
    status = record->add_u32(1, expectation.ordinal);
    if (status.ok()) {
      status = record->add_ascii_utf8(2, expectation.shard_name);
    }
    if (status.ok()) status = record->add_u64(3, identity.file_bytes);
    if (status.ok()) {
      status = record->add_u64(4, identity.filesystem_identity);
    }
    if (status.ok()) status = record->add_u64(5, identity.file_identity);
    if (status.ok()) {
      status = record->add_u64(
          6, static_cast<std::uint64_t>(identity.data_mtime_seconds));
    }
    if (status.ok()) {
      status = record->add_u32(7, identity.data_mtime_nanoseconds);
    }
    if (status.ok()) {
      status = record->add_u32(
          8, static_cast<std::uint32_t>(expectation.immutability_mode));
    }
    if (status.ok()) status = record->add_u32(9, digest_is_nonzero ? 1U : 0U);
    if (status.ok() && digest_is_nonzero) {
      status = record->add_hash(10, expectation.enforced_digest);
    }
    if (!status.ok()) return status;
    auto root = record->finalize();
    if (!root.ok()) return root.status();
    status = collection->add_hash(
        static_cast<std::uint16_t>(index + 2U), *root);
  }
  if (!status.ok()) return status;
  return collection->finalize();
}

#if !defined(PIH_DEEPSEEK_NATIVE_PLUGIN)
Result<Sha256Digest> compile_descriptor_set(
    const DeepSeekControllerArtifactCatalog& catalog,
    const DeepSeekRankMappingPlan& mapping,
    std::span<const DeepSeekWorkerShardDescriptor> descriptors,
    std::vector<DeepSeekRankArtifactDescriptorExpectation>* expectations) {
  if (descriptors.size() != mapping.shards.size() || descriptors.empty()) {
    return Status::InvalidArgument(
        "DeepSeek rank handoff descriptor count is invalid");
  }
  if (expectations == nullptr || !expectations->empty()) {
    return Status::InvalidArgument(
        "DeepSeek rank handoff descriptor output is invalid");
  }
  expectations->reserve(descriptors.size());
  std::set<std::string_view, std::less<>> names;
  for (std::size_t index = 0; index < descriptors.size(); ++index) {
    const auto& descriptor = descriptors[index];
    const auto& identity = descriptor.descriptor.identity();
    auto enforced_digest =
        catalog.shard_enforced_digest(descriptor.shard_name);
    if (!enforced_digest.ok()) return enforced_digest.status();
    const auto immutability_mode = catalog.immutability_mode();
    const auto digest_is_nonzero = nonzero(*enforced_digest);
    if (descriptor.shard_name != mapping.shards[index].shard_name ||
        identity.file_bytes != mapping.shards[index].file_bytes ||
        identity.file_bytes == 0 || identity.filesystem_identity == 0 ||
        identity.file_identity == 0 || identity.data_mtime_seconds < 0 ||
        identity.data_mtime_nanoseconds >= 1'000'000'000U ||
        (immutability_mode == ArtifactImmutabilityMode::kFsVerity &&
         !digest_is_nonzero) ||
        (immutability_mode != ArtifactImmutabilityMode::kFsVerity &&
         digest_is_nonzero) ||
        !names.insert(descriptor.shard_name).second) {
      return Status::InvalidArgument(
          "DeepSeek rank handoff descriptor identity is invalid");
    }
    expectations->push_back(
        {static_cast<std::uint32_t>(index), descriptor.shard_name, identity,
         immutability_mode, *enforced_digest});
  }
  return compile_descriptor_expectation_set(*expectations);
}
#endif

}  // namespace

Result<Sha256Digest> compile_deepseek_stage_mapping_plan_root(
    const DeepSeekStageMappingPlan& mapping) {
  auto identity = compile_mapping_identity(mapping);
  if (!identity.ok()) return identity.status();
  return identity->root;
}

Result<Sha256Digest> compile_deepseek_rank_mapping_plan_root(
    const DeepSeekRankMappingPlan& mapping) {
  return compile_rank_mapping_root_impl(mapping, mapping.rank);
}

Result<Sha256Digest> compile_deepseek_rank_tensor_handoff_root(
    std::span<const DeepSeekRankTensorRecord> records) {
  return compile_tensor_set_impl(records);
}

Result<Sha256Digest> compile_deepseek_rank_artifact_descriptor_handoff_root(
    std::span<const DeepSeekRankArtifactDescriptorExpectation>
        descriptors) {
  return compile_descriptor_expectation_set(descriptors);
}

Result<Sha256Digest> compile_deepseek_rank_artifact_handoff_manifest_root(
    std::uint32_t rank, const DeepSeekRankMappingPlan& mapping,
    std::span<const DeepSeekRankArtifactDescriptorExpectation> descriptors,
    std::span<const DeepSeekRankTensorRecord> tensor_records) {
  if (mapping.rank != rank || descriptors.size() > UINT32_MAX ||
      tensor_records.size() > UINT32_MAX) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact handoff manifest geometry is invalid");
  }
  auto mapping_root = compile_rank_mapping_root_impl(mapping, rank);
  if (!mapping_root.ok()) return mapping_root.status();
  auto descriptor_root = compile_descriptor_expectation_set(descriptors);
  if (!descriptor_root.ok()) return descriptor_root.status();
  auto tensor_root = compile_tensor_set_impl(tensor_records);
  if (!tensor_root.ok()) return tensor_root.status();
  return compile_deepseek_rank_artifact_handoff_manifest_root_from_roots(
      rank, *mapping_root, *descriptor_root, *tensor_root,
      static_cast<std::uint32_t>(descriptors.size()),
      static_cast<std::uint32_t>(tensor_records.size()));
}

Result<Sha256Digest>
compile_deepseek_rank_artifact_handoff_manifest_root_from_roots(
    std::uint32_t rank, const Sha256Digest& rank_mapping_root,
    const Sha256Digest& descriptor_root, const Sha256Digest& tensor_root,
    std::uint32_t descriptor_count, std::uint32_t tensor_record_count) {
  if (rank > 3 || rank_mapping_root == Sha256Digest{} ||
      descriptor_root == Sha256Digest{} || tensor_root == Sha256Digest{} ||
      descriptor_count == 0 ||
      descriptor_count >
          DeepSeekRuntimeArtifactManifest::kMaximumTargetMemberCount ||
      tensor_record_count == 0 ||
      tensor_record_count >
          DeepSeekRuntimeRecordsManifest::kMaximumRecordCount) {
    return Status::InvalidArgument(
        "DeepSeek rank artifact handoff root fields are invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-handoff:v1", 6);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, rank);
  if (status.ok()) status = builder->add_hash(2, rank_mapping_root);
  if (status.ok()) status = builder->add_hash(3, descriptor_root);
  if (status.ok()) status = builder->add_hash(4, tensor_root);
  if (status.ok()) status = builder->add_u32(5, descriptor_count);
  if (status.ok()) status = builder->add_u32(6, tensor_record_count);
  if (!status.ok()) return status;
  return builder->finalize();
}

DeepSeekRankArtifactHandoffManifest::DeepSeekRankArtifactHandoffManifest(
    std::uint32_t rank, DeepSeekRankMappingPlan mapping,
    std::vector<DeepSeekRankTensorRecord> tensor_records,
    std::vector<DeepSeekRankArtifactDescriptorExpectation>
        descriptor_expectations,
    Sha256Digest manifest_root) noexcept
    : rank_(rank), mapping_(std::move(mapping)),
      tensor_records_(std::move(tensor_records)),
      descriptor_expectations_(std::move(descriptor_expectations)),
      manifest_root_(manifest_root) {}

DeepSeekRankArtifactHandoffPlan::DeepSeekRankArtifactHandoffPlan(
    std::uint32_t world_size, ArtifactImmutabilityMode immutability_mode,
    bool source_catalog_production_eligible, Sha256Digest artifact_root,
    Sha256Digest mapping_root, Sha256Digest plan_root,
    std::vector<DeepSeekRankArtifactHandoffManifest> rank_manifests,
    std::vector<std::vector<DeepSeekWorkerShardDescriptor>> descriptors) noexcept
    : world_size_(world_size), immutability_mode_(immutability_mode),
      source_catalog_production_eligible_(source_catalog_production_eligible),
      artifact_root_(artifact_root),
      mapping_root_(mapping_root), plan_root_(plan_root),
      rank_manifests_(std::move(rank_manifests)),
      descriptors_(std::move(descriptors)) {}

#if !defined(PIH_DEEPSEEK_NATIVE_PLUGIN)
Result<DeepSeekRankArtifactHandoffPlan>
DeepSeekRankArtifactHandoffPlan::Compile(
    const DeepSeekControllerArtifactCatalog& catalog) {
  if (!catalog.target_authority_bound() ||
      !nonzero(catalog.artifact_root())) {
    return Status::FailedPrecondition(
        "DeepSeek artifact handoff requires a target-authoritative catalog");
  }
  auto status = catalog.poll_master_leases();
  if (!status.ok()) return status;
  auto mapping = compile_mapping_identity(catalog.mapping_plan());
  if (!mapping.ok()) return mapping.status();
  const auto world_size = catalog.mapping_plan().world_size();

  std::vector<std::vector<DeepSeekWorkerShardDescriptor>> descriptors;
  std::vector<DeepSeekRankArtifactHandoffManifest> rank_manifests;
  std::vector<Sha256Digest> rank_roots;
  descriptors.reserve(world_size);
  rank_manifests.reserve(world_size);
  rank_roots.reserve(world_size);
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    auto rank_descriptors = catalog.duplicate_rank_descriptors(rank);
    if (!rank_descriptors.ok()) return rank_descriptors.status();
    std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations;
    auto descriptor_root = compile_descriptor_set(
        catalog, catalog.mapping_plan().rank(rank), *rank_descriptors,
        &expectations);
    if (!descriptor_root.ok()) return descriptor_root.status();
    auto tensor_records = catalog.rank_tensor_records(rank);
    if (!tensor_records.ok()) return tensor_records.status();
    if (std::ranges::any_of(*tensor_records, [&](const auto& record) {
          return record.artifact_root != catalog.artifact_root();
        })) {
      return Status::InvalidArgument(
          "DeepSeek rank tensor artifact root differs from catalog");
    }
    auto tensor_root = compile_tensor_set_impl(*tensor_records);
    if (!tensor_root.ok()) return tensor_root.status();

    auto rank_root = compile_deepseek_rank_artifact_handoff_manifest_root(
        rank, catalog.mapping_plan().rank(rank), expectations,
        *tensor_records);
    if (!rank_root.ok()) return rank_root.status();
    auto rank_manifest = DeepSeekRankArtifactHandoffManifest(
        rank, catalog.mapping_plan().rank(rank),
        std::vector<DeepSeekRankTensorRecord>(tensor_records->begin(),
                                              tensor_records->end()),
        std::move(expectations), *rank_root);
    rank_manifests.push_back(std::move(rank_manifest));
    rank_roots.push_back(*rank_root);
    descriptors.push_back(std::move(*rank_descriptors));
  }

  auto plan = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-handoff-plan:v1",
      world_size + 6U);
  if (!plan.ok()) return plan.status();
  status = plan->add_u32(1, world_size);
  if (status.ok()) {
    status = plan->add_u32(
        2, static_cast<std::uint32_t>(catalog.immutability_mode()));
  }
  if (status.ok()) {
    status = plan->add_u32(3, catalog.production_eligible() ? 1U : 0U);
  }
  if (status.ok()) status = plan->add_hash(4, catalog.artifact_root());
  if (status.ok()) status = plan->add_hash(5, mapping->root);
  if (status.ok()) {
    status = plan->add_u32(
        6, static_cast<std::uint32_t>(catalog.master_shard_count()));
  }
  for (std::uint32_t rank = 0; status.ok() && rank < world_size; ++rank) {
    status = plan->add_hash(
        static_cast<std::uint16_t>(rank + 7U), rank_roots[rank]);
  }
  if (!status.ok()) return status;
  auto plan_root = plan->finalize();
  if (!plan_root.ok()) return plan_root.status();
  return DeepSeekRankArtifactHandoffPlan(
      world_size, catalog.immutability_mode(), catalog.production_eligible(),
      catalog.artifact_root(), mapping->root, *plan_root,
      std::move(rank_manifests), std::move(descriptors));
}
#endif

const Sha256Digest& DeepSeekRankArtifactHandoffPlan::rank_root(
    std::uint32_t rank) const {
  if (rank >= rank_manifests_.size()) {
    throw std::out_of_range("DeepSeek artifact handoff rank root");
  }
  return rank_manifests_[rank].manifest_root();
}

const DeepSeekRankArtifactHandoffManifest&
DeepSeekRankArtifactHandoffPlan::rank_manifest(std::uint32_t rank) const {
  if (rank >= rank_manifests_.size()) {
    throw std::out_of_range("DeepSeek artifact handoff rank manifest");
  }
  return rank_manifests_[rank];
}

std::span<const DeepSeekWorkerShardDescriptor>
DeepSeekRankArtifactHandoffPlan::descriptors(std::uint32_t rank) const {
  if (rank >= descriptors_.size()) {
    throw std::out_of_range("DeepSeek artifact handoff rank descriptors");
  }
  return descriptors_[rank];
}

}  // namespace pih
