#include "pih/model/deepseek_capability_artifact_catalog.h"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>

#include "pih/model/deepseek_runtime_artifact_manifest.h"
#include "pih/model/deepseek_runtime_records_manifest.h"
#include "pih/model/safetensors_shard_index.h"
#include "pih/model/deepseek_tensor_ownership_plan.h"

namespace pih {
namespace {

Result<std::string> ReadComplete(
    const DeepSeekVerifiedArtifactLease& lease, std::uint64_t maximum_bytes,
    std::string_view label) {
  const auto file_bytes = lease.file_bytes();
  if (file_bytes == 0 || file_bytes > maximum_bytes ||
      file_bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        std::string("DeepSeek ") + std::string(label) +
        " exceeds its capability byte budget");
  }
  std::string source(static_cast<std::size_t>(file_bytes), '\0');
  auto status = lease.read_exact(
      0, std::as_writable_bytes(
             std::span<char>(source.data(), source.size())));
  if (!status.ok()) return status;
  return source;
}

Status VerifyExactObject(const DeepSeekVerifiedArtifactLease& lease,
                         std::uint64_t expected_bytes,
                         const Sha256Digest& expected_sha256,
                         std::string_view label) {
  if (lease.file_bytes() != expected_bytes) {
    return Status::InvalidArgument(
        std::string("DeepSeek ") + std::string(label) +
        " length differs from target manifest");
  }
  constexpr std::size_t kChunkBytes = 1U << 20;
  std::vector<std::byte> buffer(kChunkBytes);
  Sha256 digest;
  std::uint64_t offset = 0;
  while (offset < expected_bytes) {
    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
        expected_bytes - offset, buffer.size()));
    auto chunk = std::span<std::byte>(buffer.data(), count);
    auto status = lease.read_exact(offset, chunk);
    if (!status.ok()) return status;
    status = digest.update(chunk);
    if (!status.ok()) return status;
    offset += count;
  }
  auto observed = digest.finalize();
  if (!observed.ok()) return observed.status();
  if (*observed != expected_sha256) {
    return Status::InvalidArgument(
        std::string("DeepSeek ") + std::string(label) +
        " SHA-256 differs from target manifest");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekCapabilityArtifactCatalog>
DeepSeekCapabilityArtifactCatalog::OpenFlash0731TargetGeneration(
    const pih_verified_artifact_api_v1& artifact_api,
    const std::filesystem::path& trusted_root,
    const Sha256Digest& expected_artifact_root,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes) {
  if (trusted_root.empty() || pipeline.world_size() != 1 ||
      maximum_shard_bytes == 0) {
    return Status::InvalidArgument(
        "DeepSeek capability target generation input is invalid");
  }
  const auto open = [&](std::string_view name, std::uint64_t maximum_bytes)
      -> Result<std::shared_ptr<DeepSeekVerifiedArtifactLease>> {
    auto lease = DeepSeekVerifiedArtifactLease::OpenDevelopment(
        artifact_api, trusted_root, name, maximum_bytes);
    if (!lease.ok()) return lease.status();
    return std::make_shared<DeepSeekVerifiedArtifactLease>(
        std::move(*lease));
  };

  auto manifest_lease = open(
      "pih.manifest.json", DeepSeekRuntimeArtifactManifest::kMaximumBytes);
  if (!manifest_lease.ok()) return manifest_lease.status();
  auto manifest_source = ReadComplete(
      **manifest_lease, DeepSeekRuntimeArtifactManifest::kMaximumBytes,
      "runtime artifact manifest");
  if (!manifest_source.ok()) return manifest_source.status();
  auto manifest = DeepSeekRuntimeArtifactManifest::Parse(
      *manifest_source, expected_artifact_root);
  if (!manifest.ok()) return manifest.status();
  if (manifest->world_size() != 1 || manifest->dspark_enabled() ||
      pipeline.rank(0).owns_dspark) {
    return Status::InvalidArgument(
        "DeepSeek capability artifact profile is not PP1");
  }
  auto status = manifest->validate_flash_0731_geometry();
  if (!status.ok()) return status;

  auto index_lease = open(manifest->index().name,
                          manifest->index().file_bytes);
  if (!index_lease.ok()) return index_lease.status();
  status = VerifyExactObject(**index_lease, manifest->index().file_bytes,
                             manifest->index().object_sha256,
                             "target index");
  if (!status.ok()) return status;
  auto index_source = ReadComplete(
      **index_lease, SafetensorsShardIndex::kMaximumIndexBytes,
      "target index");
  if (!index_source.ok()) return index_source.status();
  auto index = SafetensorsShardIndex::Parse(*index_source);
  if (!index.ok()) return index.status();
  if (index->total_size() != manifest->tensor_bytes() ||
      index->bindings().size() != manifest->tensor_count()) {
    return Status::InvalidArgument(
        "DeepSeek capability index ledger differs from manifest");
  }

  auto records_lease = open("pih.runtime-records.json",
                            manifest->runtime_records().object_bytes);
  if (!records_lease.ok()) return records_lease.status();
  status = VerifyExactObject(**records_lease,
                             manifest->runtime_records().object_bytes,
                             manifest->runtime_records().object_sha256,
                             "runtime records");
  if (!status.ok()) return status;
  auto records_source = ReadComplete(
      **records_lease, DeepSeekRuntimeRecordsManifest::kMaximumBytes,
      "runtime records");
  if (!records_source.ok()) return records_source.status();
  auto runtime_records = DeepSeekRuntimeRecordsManifest::Parse(
      *records_source, manifest->runtime_records(), pipeline);
  if (!runtime_records.ok()) return runtime_records.status();

  std::map<std::string, const DeepSeekRuntimeArtifactShard*, std::less<>>
      expected_shards;
  std::vector<std::string> expected_names;
  expected_names.reserve(manifest->shards().size());
  for (const auto& shard : manifest->shards()) {
    expected_names.push_back(shard.shard_name);
    if (!expected_shards.emplace(shard.shard_name, &shard).second) {
      return Status::InvalidArgument(
          "DeepSeek capability manifest shard is duplicated");
    }
  }
  std::ranges::sort(expected_names);
  if (expected_names != index->shard_names()) {
    return Status::InvalidArgument(
        "DeepSeek capability index shard set differs from manifest");
  }
  if (index->bindings().size() != runtime_records->records().size()) {
    return Status::InvalidArgument(
        "DeepSeek capability record count differs from index");
  }
  for (std::size_t ordinal = 0; ordinal < index->bindings().size();
       ++ordinal) {
    const auto& binding = index->bindings()[ordinal];
    const auto& record = runtime_records->records()[ordinal];
    if (binding.tensor_name != record.tensor_name ||
        binding.shard_name != record.shard_name) {
      return Status::InvalidArgument(
          "DeepSeek capability index projection differs from records");
    }
  }

  DeepSeekCapabilityArtifactCatalog result;
  result.artifact_root_ = manifest->artifact_root();
  result.authority_leases_.push_back(std::move(*manifest_lease));
  result.authority_leases_.push_back(std::move(*index_lease));
  result.authority_leases_.push_back(std::move(*records_lease));
  result.shards_.reserve(manifest->shards().size());
  for (const auto& expected : manifest->shards()) {
    if (expected.file_bytes > maximum_shard_bytes) {
      return Status::ResourceExhausted(
          "DeepSeek capability shard exceeds byte budget");
    }
    auto lease = open(expected.shard_name, maximum_shard_bytes);
    if (!lease.ok()) return lease.status();
    status = VerifyExactObject(**lease, expected.file_bytes,
                               expected.object_sha256, "target shard");
    if (!status.ok()) return status;
    auto receipt = load_safetensors_header_capability(**lease);
    if (!receipt.ok()) return receipt.status();
    if (receipt->file_bytes != expected.file_bytes ||
        receipt->header.tensors().size() != expected.tensor_count ||
        receipt->header.data_bytes() != expected.payload_bytes) {
      return Status::InvalidArgument(
          "DeepSeek capability shard header differs from manifest");
    }
    result.shards_.push_back(
        {expected.shard_name, std::move(*lease), std::move(*receipt)});
  }

  std::map<std::string, std::uint32_t, std::less<>> observed_per_shard;
  for (const auto& record : runtime_records->records()) {
    const auto expected = expected_shards.find(record.shard_name);
    const auto* header = result.header(record.shard_name);
    const auto* tensor = header == nullptr
                             ? nullptr
                             : header->tensor(record.tensor_name);
    if (expected == expected_shards.end() || tensor == nullptr ||
        tensor->dtype != record.dtype || tensor->shape != record.shape ||
        tensor->file_begin != record.file_begin ||
        tensor->file_end != record.file_end ||
        record.tensor_bytes != record.file_end - record.file_begin) {
      return Status::InvalidArgument(
          "DeepSeek capability shard header differs from runtime record");
    }
    ++observed_per_shard[record.shard_name];
  }
  for (const auto& expected : manifest->shards()) {
    if (observed_per_shard[expected.shard_name] != expected.tensor_count) {
      return Status::InvalidArgument(
          "DeepSeek capability runtime record closure is incomplete");
    }
  }

  auto ownership = DeepSeekTensorOwnershipPlan::BindTargetManifest(
      pipeline, *runtime_records);
  if (!ownership.ok()) return ownership.status();
  std::vector<DeepSeekShardHeaderView> views;
  views.reserve(result.shards_.size());
  for (const auto& shard : result.shards_) {
    views.push_back(
        {shard.name, shard.receipt.file_bytes, &shard.receipt.header});
  }
  auto mapping = DeepSeekStageMappingPlan::BindTargetManifest(
      *ownership, views, DeepSeekStageMappingPlan::kProductionPageBytes);
  if (!mapping.ok()) return mapping.status();
  result.mapping_plan_ = std::move(*mapping);
  result.rank_tensors_.resize(1);
  for (const auto& record : runtime_records->records()) {
    result.rank_tensors_[0].push_back(
        {record.tensor_name,
         record.shard_name,
         record.role,
         record.dtype,
         record.shape,
         record.file_begin,
         record.file_end,
         record.logical_layer,
         record.tensor_bytes,
         record.storage_semantics,
         manifest->artifact_root(),
         manifest->layout_root(),
         manifest->disposition_root(),
         record.target_logical_root,
         record.disposition_record_root,
         record.layout_record_root,
         record.runtime_record_root});
  }
  if (result.rank_tensors_[0].size() != ownership->owned_count(0) ||
      result.rank_tensors_[0].size() !=
          result.mapping_plan_.rank(0).owned_tensor_count) {
    return Status::Internal(
        "DeepSeek capability ownership publication is incomplete");
  }
  status = result.poll_leases();
  if (!status.ok()) return status;
  return result;
}

const SafetensorsHeader* DeepSeekCapabilityArtifactCatalog::header(
    std::string_view shard_name) const noexcept {
  const auto found = std::ranges::find(
      shards_, shard_name, &DeepSeekCapabilityArtifactCatalog::Shard::name);
  return found == shards_.end() ? nullptr : &found->receipt.header;
}

Result<std::vector<DeepSeekCapabilityShardLease>>
DeepSeekCapabilityArtifactCatalog::rank_leases(std::uint32_t rank) const {
  if (rank != 0 || mapping_plan_.world_size() != 1) {
    return Status::InvalidArgument(
        "DeepSeek capability rank is outside PP1 catalog");
  }
  std::vector<DeepSeekCapabilityShardLease> leases;
  leases.reserve(mapping_plan_.rank(0).shards.size());
  for (const auto& receipt : mapping_plan_.rank(0).shards) {
    const auto found = std::ranges::find(
        shards_, receipt.shard_name,
        &DeepSeekCapabilityArtifactCatalog::Shard::name);
    if (found == shards_.end()) {
      return Status::Internal(
          "DeepSeek capability rank shard is absent from catalog");
    }
    leases.push_back({found->name, found->lease});
  }
  return leases;
}

Result<std::span<const DeepSeekRankTensorRecord>>
DeepSeekCapabilityArtifactCatalog::rank_tensor_records(
    std::uint32_t rank) const {
  if (rank >= rank_tensors_.size()) {
    return Status::InvalidArgument(
        "DeepSeek capability tensor rank is outside catalog");
  }
  return std::span<const DeepSeekRankTensorRecord>(rank_tensors_[rank]);
}

Status DeepSeekCapabilityArtifactCatalog::poll_leases() const {
  for (const auto& lease : authority_leases_) {
    auto status = lease->poll_identity_unchanged();
    if (!status.ok()) return status;
  }
  for (const auto& shard : shards_) {
    auto status = shard.lease->poll_identity_unchanged();
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace pih
