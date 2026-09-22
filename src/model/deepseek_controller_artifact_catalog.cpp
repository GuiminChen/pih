#include "pih/model/deepseek_controller_artifact_catalog.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>

#include "pih/core/checked_math.h"

#if !defined(PIH_DEEPSEEK_NATIVE_PLUGIN)

namespace pih {
namespace {

Result<std::string> read_complete_authority(
    const ControllerFileLease& lease, std::uint64_t maximum_bytes,
    std::string_view label) {
  const auto file_bytes = lease.identity().file_bytes;
  if (file_bytes == 0 || file_bytes > maximum_bytes ||
      file_bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        std::string("DeepSeek ") + std::string(label) +
        " exceeds its authority byte budget");
  }
  std::string source(static_cast<std::size_t>(file_bytes), '\0');
  const auto read = lease.read_exact(
      0, std::as_writable_bytes(
             std::span<char>(source.data(), source.size())));
  if (!read.ok()) return read;
  return source;
}

Status verify_exact_object(const ControllerFileLease& lease,
                           std::uint64_t expected_bytes,
                           const Sha256Digest& expected_sha256,
                           std::string_view label) {
  if (lease.identity().file_bytes != expected_bytes) {
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
    const auto chunk = std::span<std::byte>(buffer.data(), count);
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

Result<DeepSeekControllerArtifactCatalog>
DeepSeekControllerArtifactCatalog::OpenFlash0731(
    const std::filesystem::path& trusted_root,
    const SafetensorsShardIndex& index,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes,
    ArtifactImmutabilityMode immutability_mode) {
  const auto geometry = index.validate_deepseek_flash_0731_repository_geometry();
  if (!geometry.ok()) return geometry;
  return Open(trusted_root, index, pipeline, maximum_shard_bytes,
              immutability_mode);
}

Result<DeepSeekControllerArtifactCatalog>
DeepSeekControllerArtifactCatalog::OpenFlash0731FsVerity(
    const std::filesystem::path& trusted_root,
    const SafetensorsShardIndex& index,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes,
    std::span<const DeepSeekShardFsVerityDigest> expected_digests) {
  const auto geometry = index.validate_deepseek_flash_0731_repository_geometry();
  if (!geometry.ok()) return geometry;
  return OpenImpl(trusted_root, index, pipeline, maximum_shard_bytes,
                  ArtifactImmutabilityMode::kFsVerity, expected_digests);
}

Result<DeepSeekControllerArtifactCatalog>
DeepSeekControllerArtifactCatalog::OpenFlash0731DmVerity(
    const std::filesystem::path& trusted_root,
    const SafetensorsShardIndex& index,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes,
    const DmVeritySupervisorReceipt& supervisor_receipt,
    const Sha256Digest& expected_root_digest,
    const Sha256Digest& expected_table_digest,
    const Sha256Digest& expected_supervisor_attestation_digest,
    std::uint64_t integrity_reserve_bytes) {
  const auto geometry = index.validate_deepseek_flash_0731_repository_geometry();
  if (!geometry.ok()) return geometry;
  return OpenImpl(
      trusted_root, index, pipeline, maximum_shard_bytes,
      ArtifactImmutabilityMode::kDmVeritySnapshot, {}, &supervisor_receipt,
      &expected_root_digest, &expected_table_digest,
      &expected_supervisor_attestation_digest, integrity_reserve_bytes);
}

Result<DeepSeekControllerArtifactCatalog>
DeepSeekControllerArtifactCatalog::Open(
    const std::filesystem::path& trusted_root,
    const SafetensorsShardIndex& index,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes,
    ArtifactImmutabilityMode immutability_mode) {
  return OpenImpl(trusted_root, index, pipeline, maximum_shard_bytes,
                  immutability_mode, {});
}

Result<DeepSeekControllerArtifactCatalog>
DeepSeekControllerArtifactCatalog::OpenTargetGeneration(
    const std::filesystem::path& trusted_root,
    const Sha256Digest& expected_artifact_root,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes,
    ArtifactImmutabilityMode immutability_mode) {
  return OpenTargetImpl(trusted_root, expected_artifact_root, pipeline,
                        maximum_shard_bytes, immutability_mode, false, {});
}

Result<DeepSeekControllerArtifactCatalog>
DeepSeekControllerArtifactCatalog::OpenFlash0731TargetGeneration(
    const std::filesystem::path& trusted_root,
    const Sha256Digest& expected_artifact_root,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes,
    ArtifactImmutabilityMode immutability_mode) {
  return OpenTargetImpl(trusted_root, expected_artifact_root, pipeline,
                        maximum_shard_bytes, immutability_mode, true, {});
}

Result<DeepSeekControllerArtifactCatalog>
DeepSeekControllerArtifactCatalog::OpenFlash0731TargetGenerationFsVerity(
    const std::filesystem::path& trusted_root,
    const Sha256Digest& expected_artifact_root,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes,
    std::span<const DeepSeekTargetMemberFsVerityDigest> expected_digests) {
  return OpenTargetImpl(trusted_root, expected_artifact_root, pipeline,
                        maximum_shard_bytes,
                        ArtifactImmutabilityMode::kFsVerity, true,
                        expected_digests);
}

Result<DeepSeekControllerArtifactCatalog>
DeepSeekControllerArtifactCatalog::OpenTargetImpl(
    const std::filesystem::path& trusted_root,
    const Sha256Digest& expected_artifact_root,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes,
    ArtifactImmutabilityMode immutability_mode,
    bool require_flash_0731_geometry,
    std::span<const DeepSeekTargetMemberFsVerityDigest> expected_digests) {
  if (trusted_root.empty() || pipeline.world_size() == 0 ||
      pipeline.world_size() > 4 || maximum_shard_bytes == 0) {
    return Status::InvalidArgument(
        "DeepSeek target generation catalog input is invalid");
  }
  if (immutability_mode != ArtifactImmutabilityMode::kUncalibrated &&
      immutability_mode != ArtifactImmutabilityMode::kFsVerity) {
    return Status::FailedPrecondition(
        "DeepSeek target generation immutability mode is unsupported");
  }

  std::map<std::string, const Sha256Digest*, std::less<>> verity_digests;
  if (expected_digests.size() > kMaximumTargetMemberCount) {
    return Status::ResourceExhausted(
        "DeepSeek target fs-verity digest manifest exceeds member budget");
  }
  for (const auto& expected : expected_digests) {
    const bool nonzero_digest = std::ranges::any_of(
        expected.expected_digest.bytes,
        [](std::byte byte) { return byte != std::byte{0}; });
    if (expected.member_name.empty() || expected.member_name.size() > 255 ||
        !nonzero_digest ||
        !verity_digests.emplace(expected.member_name,
                                &expected.expected_digest).second) {
      return Status::InvalidArgument(
          "DeepSeek target fs-verity digest manifest is invalid or duplicated");
    }
  }
  if ((immutability_mode == ArtifactImmutabilityMode::kFsVerity &&
       verity_digests.empty()) ||
      (immutability_mode != ArtifactImmutabilityMode::kFsVerity &&
       !verity_digests.empty())) {
    return Status::InvalidArgument(
        "DeepSeek target fs-verity authority is missing or unexpected");
  }

  const auto open_member = [&]<typename Size>(
                               std::string_view name, Size maximum_bytes)
      -> Result<ControllerFileLease> {
    if (immutability_mode == ArtifactImmutabilityMode::kFsVerity) {
      const auto expected = verity_digests.find(std::string(name));
      if (expected == verity_digests.end()) {
        return Status::InvalidArgument(
            "DeepSeek target fs-verity digest set is missing a member");
      }
      return ControllerFileLease::OpenBeneathFsVerity(
          trusted_root, name, static_cast<std::uint64_t>(maximum_bytes),
          *expected->second);
    }
    return ControllerFileLease::OpenBeneath(
        trusted_root, name, static_cast<std::uint64_t>(maximum_bytes),
        immutability_mode);
  };

  auto manifest_lease = open_member(
      "pih.manifest.json",
      DeepSeekRuntimeArtifactManifest::kMaximumBytes);
  if (!manifest_lease.ok()) return manifest_lease.status();
  if (immutability_mode == ArtifactImmutabilityMode::kFsVerity &&
      !manifest_lease->production_eligible()) {
    return Status::FailedPrecondition(
        "DeepSeek target manifest is not integrity eligible");
  }
  auto manifest_source = read_complete_authority(
      *manifest_lease, DeepSeekRuntimeArtifactManifest::kMaximumBytes,
      "runtime artifact manifest");
  if (!manifest_source.ok()) return manifest_source.status();
  auto manifest = DeepSeekRuntimeArtifactManifest::Parse(
      *manifest_source, expected_artifact_root);
  if (!manifest.ok()) return manifest.status();
  if (manifest->world_size() != pipeline.world_size() ||
      manifest->dspark_enabled() !=
          pipeline.rank(pipeline.world_size() - 1).owns_dspark) {
    return Status::InvalidArgument(
        "DeepSeek target artifact profile differs from pipeline");
  }
  if (require_flash_0731_geometry) {
    const auto geometry = manifest->validate_flash_0731_geometry();
    if (!geometry.ok()) return geometry;
  }

  std::vector<std::string> expected_members{
      "pih.manifest.json", manifest->index().name,
      "pih.runtime-records.json"};
  expected_members.reserve(3 + manifest->shards().size());
  for (const auto& shard : manifest->shards()) {
    expected_members.push_back(shard.shard_name);
  }
  std::ranges::sort(expected_members);
  if (std::adjacent_find(expected_members.begin(), expected_members.end()) !=
      expected_members.end()) {
    return Status::InvalidArgument(
        "DeepSeek target generation member name is duplicated");
  }
  if (immutability_mode == ArtifactImmutabilityMode::kFsVerity) {
    std::vector<std::string> supplied_members;
    supplied_members.reserve(verity_digests.size());
    for (const auto& [name, digest] : verity_digests) {
      (void)digest;
      supplied_members.push_back(name);
    }
    if (supplied_members != expected_members) {
      return Status::InvalidArgument(
          "DeepSeek target fs-verity digest set differs from generation");
    }
  }

  auto index_lease = open_member(manifest->index().name,
                                 manifest->index().file_bytes);
  if (!index_lease.ok()) return index_lease.status();
  if (immutability_mode == ArtifactImmutabilityMode::kFsVerity &&
      !index_lease->production_eligible()) {
    return Status::FailedPrecondition(
        "DeepSeek target index is not integrity eligible");
  }
  auto status = verify_exact_object(
      *index_lease, manifest->index().file_bytes,
      manifest->index().object_sha256, "target index");
  if (!status.ok()) return status;
  auto index_source = read_complete_authority(
      *index_lease, SafetensorsShardIndex::kMaximumIndexBytes,
      "target index");
  if (!index_source.ok()) return index_source.status();
  auto index = SafetensorsShardIndex::Parse(*index_source);
  if (!index.ok()) return index.status();
  if (index->total_size() != manifest->tensor_bytes() ||
      index->bindings().size() != manifest->tensor_count()) {
    return Status::InvalidArgument(
        "DeepSeek target index ledger differs from artifact manifest");
  }

  auto records_lease = open_member(
      "pih.runtime-records.json",
      manifest->runtime_records().object_bytes);
  if (!records_lease.ok()) return records_lease.status();
  if (immutability_mode == ArtifactImmutabilityMode::kFsVerity &&
      !records_lease->production_eligible()) {
    return Status::FailedPrecondition(
        "DeepSeek target records are not integrity eligible");
  }
  status = verify_exact_object(
      *records_lease, manifest->runtime_records().object_bytes,
      manifest->runtime_records().object_sha256, "runtime records");
  if (!status.ok()) return status;
  auto records_source = read_complete_authority(
      *records_lease, DeepSeekRuntimeRecordsManifest::kMaximumBytes,
      "runtime records");
  if (!records_source.ok()) return records_source.status();
  auto runtime_records = DeepSeekRuntimeRecordsManifest::Parse(
      *records_source, manifest->runtime_records(), pipeline);
  if (!runtime_records.ok()) return runtime_records.status();

  std::vector<std::string> manifest_shard_names;
  manifest_shard_names.reserve(manifest->shards().size());
  std::map<std::string, const DeepSeekRuntimeArtifactShard*, std::less<>>
      manifest_shards;
  for (const auto& shard : manifest->shards()) {
    manifest_shard_names.push_back(shard.shard_name);
    if (!manifest_shards.emplace(shard.shard_name, &shard).second) {
      return Status::InvalidArgument(
          "DeepSeek target manifest shard name is duplicated");
    }
  }
  std::ranges::sort(manifest_shard_names);
  if (manifest_shard_names != index->shard_names()) {
    return Status::InvalidArgument(
        "DeepSeek target index shard set differs from artifact manifest");
  }
  if (index->bindings().size() != runtime_records->records().size()) {
    return Status::InvalidArgument(
        "DeepSeek target index and runtime record counts differ");
  }
  for (std::size_t ordinal = 0; ordinal < index->bindings().size();
       ++ordinal) {
    const auto& binding = index->bindings()[ordinal];
    const auto& record = runtime_records->records()[ordinal];
    if (binding.tensor_name != record.tensor_name ||
        binding.shard_name != record.shard_name) {
      return Status::InvalidArgument(
          "DeepSeek target index projection differs from runtime records");
    }
  }

  DeepSeekControllerArtifactCatalog result;
  result.immutability_mode_ = immutability_mode;
  result.target_authority_bound_ = true;
  result.production_eligible_ =
      immutability_mode == ArtifactImmutabilityMode::kFsVerity;
  result.artifact_root_ = manifest->artifact_root();
  result.authority_files_.reserve(3);
  result.authority_files_.emplace_back(
      "pih.manifest.json", std::move(*manifest_lease));
  result.authority_files_.emplace_back(
      manifest->index().name, std::move(*index_lease));
  result.authority_files_.emplace_back(
      "pih.runtime-records.json", std::move(*records_lease));
  result.shards_.reserve(manifest->shards().size());
  for (const auto& expected : manifest->shards()) {
    if (expected.file_bytes > maximum_shard_bytes) {
      return Status::ResourceExhausted(
          "DeepSeek target shard exceeds catalog byte budget");
    }
    auto lease = open_member(expected.shard_name, maximum_shard_bytes);
    if (!lease.ok()) return lease.status();
    if (result.production_eligible_ && !lease->production_eligible()) {
      return Status::FailedPrecondition(
          "DeepSeek target member is not integrity eligible");
    }
    status = verify_exact_object(*lease, expected.file_bytes,
                                 expected.object_sha256, "target shard");
    if (!status.ok()) return status;
    auto receipt = load_safetensors_header_descriptor(*lease);
    if (!receipt.ok()) return receipt.status();
    if (receipt->file_bytes != expected.file_bytes ||
        receipt->header.tensors().size() != expected.tensor_count ||
        receipt->header.data_bytes() != expected.payload_bytes) {
      return Status::InvalidArgument(
          "DeepSeek target shard header differs from artifact manifest");
    }
    result.shards_.emplace_back(expected.shard_name, std::move(*lease),
                                std::move(*receipt));
  }

  std::map<std::string, std::uint32_t, std::less<>> observed_per_shard;
  for (const auto& record : runtime_records->records()) {
    const auto expected_shard = manifest_shards.find(record.shard_name);
    const auto* header = result.header(record.shard_name);
    const auto* tensor = header == nullptr
                             ? nullptr
                             : header->tensor(record.tensor_name);
    if (expected_shard == manifest_shards.end() || tensor == nullptr ||
        tensor->dtype != record.dtype || tensor->shape != record.shape ||
        tensor->file_begin != record.file_begin ||
        tensor->file_end != record.file_end ||
        record.tensor_bytes != record.file_end - record.file_begin) {
      return Status::InvalidArgument(
          "DeepSeek target shard header differs from runtime record");
    }
    ++observed_per_shard[record.shard_name];
  }
  for (const auto& expected : manifest->shards()) {
    if (observed_per_shard[expected.shard_name] != expected.tensor_count) {
      return Status::InvalidArgument(
          "DeepSeek target shard runtime record closure is incomplete");
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
  result.rank_tensors_.resize(pipeline.world_size());
  for (const auto& record : runtime_records->records()) {
    result.rank_tensors_[record.owner_rank].push_back(
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
  std::uint64_t published_records = 0;
  for (std::uint32_t rank = 0; rank < pipeline.world_size(); ++rank) {
    if (result.rank_tensors_[rank].size() != ownership->owned_count(rank) ||
        result.rank_tensors_[rank].size() !=
            result.mapping_plan_.rank(rank).owned_tensor_count) {
      return Status::Internal(
          "DeepSeek target ownership publication is incomplete");
    }
    published_records += result.rank_tensors_[rank].size();
  }
  if (published_records != runtime_records->records().size()) {
    return Status::Internal(
        "DeepSeek target runtime record publication is incomplete");
  }
  status = result.poll_master_leases();
  if (!status.ok()) return status;
  return result;
}

Result<DeepSeekControllerArtifactCatalog>
DeepSeekControllerArtifactCatalog::OpenImpl(
    const std::filesystem::path& trusted_root,
    const SafetensorsShardIndex& index,
    const DeepSeekPipelinePlan& pipeline,
    std::uint64_t maximum_shard_bytes,
    ArtifactImmutabilityMode immutability_mode,
    std::span<const DeepSeekShardFsVerityDigest> expected_digests,
    const DmVeritySupervisorReceipt* dm_receipt,
    const Sha256Digest* expected_dm_root,
    const Sha256Digest* expected_dm_table,
    const Sha256Digest* expected_dm_attestation,
    std::uint64_t integrity_reserve_bytes) {
  if (trusted_root.empty() || index.shard_names().empty() ||
      maximum_shard_bytes == 0) {
    return Status::InvalidArgument(
        "DeepSeek controller artifact catalog input is invalid");
  }
  std::map<std::string, const Sha256Digest*, std::less<>> digests;
  for (const auto& expected : expected_digests) {
    if (expected.shard_name.empty() ||
        !digests.emplace(expected.shard_name,
                         &expected.expected_digest).second) {
      return Status::InvalidArgument(
          "DeepSeek fs-verity digest manifest is invalid or duplicated");
    }
  }
  if ((immutability_mode == ArtifactImmutabilityMode::kFsVerity &&
       digests.size() != index.shard_names().size()) ||
      (immutability_mode != ArtifactImmutabilityMode::kFsVerity &&
       !digests.empty())) {
    return Status::InvalidArgument(
        "DeepSeek fs-verity digest set differs from shard index");
  }
  const bool dm_authority_complete =
      dm_receipt != nullptr && expected_dm_root != nullptr &&
      expected_dm_table != nullptr && expected_dm_attestation != nullptr &&
      integrity_reserve_bytes != 0;
  if ((immutability_mode == ArtifactImmutabilityMode::kDmVeritySnapshot &&
       !dm_authority_complete) ||
      (immutability_mode != ArtifactImmutabilityMode::kDmVeritySnapshot &&
       (dm_receipt != nullptr || expected_dm_root != nullptr ||
        expected_dm_table != nullptr || expected_dm_attestation != nullptr ||
        integrity_reserve_bytes != 0))) {
    return Status::InvalidArgument(
        "DeepSeek dm-verity snapshot authority is incomplete or unexpected");
  }
  DeepSeekControllerArtifactCatalog result;
  result.immutability_mode_ = immutability_mode;
  result.shards_.reserve(index.shard_names().size());
  std::uint64_t payload_bytes = 0;
  for (const auto& shard_name : index.shard_names()) {
    Result<ControllerFileLease> lease =
        Status::Internal("DeepSeek artifact lease opener was not selected");
    if (immutability_mode == ArtifactImmutabilityMode::kFsVerity) {
      const auto expected = digests.find(shard_name);
      if (expected == digests.end()) {
        return Status::InvalidArgument(
            "DeepSeek fs-verity digest set is missing a shard");
      }
      lease = ControllerFileLease::OpenBeneathFsVerity(
          trusted_root, shard_name, maximum_shard_bytes, *expected->second);
    } else if (immutability_mode ==
               ArtifactImmutabilityMode::kDmVeritySnapshot) {
      lease = ControllerFileLease::OpenBeneathDmVeritySnapshot(
          trusted_root, shard_name, maximum_shard_bytes, *dm_receipt,
          *expected_dm_root, *expected_dm_table, *expected_dm_attestation,
          integrity_reserve_bytes);
    } else {
      lease = ControllerFileLease::OpenBeneath(
          trusted_root, shard_name, maximum_shard_bytes, immutability_mode);
    }
    if (!lease.ok()) return lease.status();
    if (immutability_mode != ArtifactImmutabilityMode::kUncalibrated &&
        !lease->production_eligible()) {
      return Status::FailedPrecondition(
          "DeepSeek production shard lease is not integrity eligible");
    }
    if (immutability_mode ==
        ArtifactImmutabilityMode::kDmVeritySnapshot) {
      if (result.integrity_owner_bytes_ != 0 &&
          result.integrity_owner_bytes_ != lease->integrity_owner_bytes()) {
        return Status::FailedPrecondition(
            "DeepSeek dm-verity shards disagree on integrity owner");
      }
      result.integrity_owner_bytes_ = lease->integrity_owner_bytes();
    }
    auto receipt = load_safetensors_header_descriptor(*lease);
    if (!receipt.ok()) return receipt.status();
    auto total = checked_add_u64(payload_bytes, receipt->header.data_bytes());
    if (!total.ok()) return total.status();
    payload_bytes = *total;
    result.shards_.emplace_back(shard_name, std::move(*lease),
                                std::move(*receipt));
  }
  if (payload_bytes != index.total_size()) {
    return Status::InvalidArgument(
        "DeepSeek index total_size differs from shard payload bytes");
  }
  std::vector<DeepSeekShardHeaderView> views;
  views.reserve(result.shards_.size());
  for (const auto& shard : result.shards_) {
    views.push_back({shard.name, shard.receipt.file_bytes,
                     &shard.receipt.header});
  }
  auto mapping_plan = DeepSeekStageMappingPlan::Create(
      pipeline, index.bindings(), views,
      DeepSeekStageMappingPlan::kProductionPageBytes);
  if (!mapping_plan.ok()) return mapping_plan.status();
  result.mapping_plan_ = std::move(*mapping_plan);
  auto ownership = DeepSeekTensorOwnershipPlan::Create(pipeline, index.bindings());
  if (!ownership.ok()) return ownership.status();
  result.rank_tensors_.resize(pipeline.world_size());
  for (const auto& owned : ownership->records()) {
    if (owned.owner_rank == DeepSeekTensorOwnership::kExcludedRank) continue;
    const auto* header = result.header(owned.shard_name);
    const auto* tensor = header == nullptr ? nullptr : header->tensor(owned.tensor_name);
    if (tensor == nullptr) {
      return Status::Internal("DeepSeek owned tensor disappeared from master header");
    }
    result.rank_tensors_[owned.owner_rank].push_back(
        {owned.tensor_name, owned.shard_name, owned.role, tensor->dtype,
         tensor->shape, tensor->file_begin, tensor->file_end});
  }
  return result;
}

const SafetensorsHeader* DeepSeekControllerArtifactCatalog::header(
    std::string_view shard_name) const noexcept {
  for (const auto& shard : shards_) {
    if (shard.name == shard_name) return &shard.receipt.header;
  }
  return nullptr;
}

Result<Sha256Digest>
DeepSeekControllerArtifactCatalog::shard_enforced_digest(
    std::string_view shard_name) const {
  if (shard_name.empty()) {
    return Status::InvalidArgument("DeepSeek shard name is empty");
  }
  for (const auto& shard : shards_) {
    if (shard.name == shard_name) return shard.lease.enforced_digest();
  }
  return Status::InvalidArgument("DeepSeek artifact shard is unknown");
}

Result<std::vector<DeepSeekWorkerShardDescriptor>>
DeepSeekControllerArtifactCatalog::duplicate_rank_descriptors(
    std::uint32_t rank) const {
  if (rank >= mapping_plan_.world_size()) {
    return Status::InvalidArgument("DeepSeek artifact rank is out of range");
  }
  const auto stable = poll_master_leases();
  if (!stable.ok()) return stable;
  const auto& rank_plan = mapping_plan_.rank(rank);
  std::vector<DeepSeekWorkerShardDescriptor> result;
  result.reserve(rank_plan.shards.size());
  for (const auto& required : rank_plan.shards) {
    const Shard* master = nullptr;
    for (const auto& shard : shards_) {
      if (shard.name == required.shard_name) {
        master = &shard;
        break;
      }
    }
    if (master == nullptr) {
      return Status::Internal(
          "DeepSeek rank plan references an absent master shard");
    }
    auto duplicate = master->lease.duplicate_for_worker();
    if (!duplicate.ok()) return duplicate.status();
    result.emplace_back(required.shard_name, std::move(*duplicate));
  }
  return result;
}

Result<std::span<const DeepSeekRankTensorRecord>>
DeepSeekControllerArtifactCatalog::rank_tensor_records(
    std::uint32_t rank) const {
  if (rank >= rank_tensors_.size()) {
    return Status::InvalidArgument("DeepSeek artifact rank is out of range");
  }
  return std::span<const DeepSeekRankTensorRecord>(rank_tensors_[rank]);
}

Status DeepSeekControllerArtifactCatalog::poll_master_leases(
    const DmVeritySupervisorReceipt* current_dm_receipt) const {
  for (const auto& authority : authority_files_) {
    const auto polled =
        authority.lease.poll_integrity_unchanged(current_dm_receipt);
    if (!polled.ok()) {
      return Status::FailedPrecondition(
          "DeepSeek target authority lease failed epoch poll");
    }
  }
  for (const auto& shard : shards_) {
    const auto polled =
        shard.lease.poll_integrity_unchanged(current_dm_receipt);
    if (!polled.ok()) {
      return Status::FailedPrecondition(
          "DeepSeek artifact master lease failed epoch poll");
    }
  }
  return Status::Ok();
}

}  // namespace pih

#endif  // !defined(PIH_DEEPSEEK_NATIVE_PLUGIN)
