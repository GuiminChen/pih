#include "generation_copy.h"
#include <map>
#include <set>
#include <sys/stat.h>

namespace pih::offline_deepseek {
namespace {
bool SameSource(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
      a.st_size == b.st_size && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
      a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
      a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
}  // namespace
Result<CopiedGeneration> CopyGenerationShards(int staging_directory_fd,
    std::span<const int> admitted_source_descriptors,
    std::span<const IdentityTensor> tensors,
    std::span<const TensorLayoutAuthority> layout_authorities,
    std::span<const TensorDispositionAuthority> disposition_authorities,
    const GenerationLayoutAuthority& authority,
    const Sha256Digest& converter_identity_root) {
  if (converter_identity_root == Sha256Digest{} || admitted_source_descriptors.size() != 50)
    return Status::InvalidArgument("generation converter identity missing");
  auto metadata = BuildGenerationMetadata(tensors, layout_authorities,
                                         disposition_authorities, authority);
  if (!metadata.ok()) return metadata.status();
  struct stat directory{};
  if (::fstat(staging_directory_fd, &directory) != 0 || !S_ISDIR(directory.st_mode))
    return Status::InvalidArgument("generation staging descriptor is not a directory");
  std::map<int, struct stat> sources;
  std::set<std::pair<dev_t, ino_t>> identities;
  for (const auto fd : admitted_source_descriptors) {
    struct stat source{};
    if (fd < 0 || sources.contains(fd) || ::fstat(fd, &source) != 0 ||
        !S_ISREG(source.st_mode) || source.st_size < 0 ||
        !identities.emplace(source.st_dev, source.st_ino).second)
      return Status::InvalidArgument("generation source descriptors invalid, aliased or excessive");
    sources.emplace(fd, source);
  }
  if (sources.size() != 50)
    return Status::InvalidArgument("generation source count differs from frozen converter contract");
  // Validate all ranges before creating the first shard, not only when their
  // particular shard is reached. Per-payload digest checking occurs on copy.
  for (const auto& tensor : tensors) {
    const auto& range = tensor.source;
    if (!sources.contains(range.source_fd))
      return Status::InvalidArgument("tensor source is outside admitted descriptor set");
    const auto bytes = static_cast<std::uint64_t>(sources.at(range.source_fd).st_size);
    if (range.begin > bytes || range.bytes > bytes - range.begin)
      return Status::InvalidArgument("generation source range outside admitted descriptor");
  }
  DeepSeekRuntimeArtifactManifest::EncodeInput manifest;
  manifest.logical_layout_root = metadata->bound_layout.logical_layout_root;
  manifest.source_inventory_root = authority.source_inventory_root;
  manifest.source_payload_closure_root = authority.source_payload_closure_root;
  manifest.converter_identity_root = converter_identity_root;
  manifest.records = metadata->runtime_records.authority;
  manifest.maximum_copy_chunk_bytes = 1U << 20;
  manifest.source_descriptor_count = static_cast<std::uint32_t>(sources.size());
  manifest.maximum_output_descriptors = 1;
  const auto& layout = metadata->bound_layout.layout;
  manifest.index = {"model.safetensors.index.json", layout.index_json.size(),
                    layout.index_sha256, metadata->bound_layout.index_root, {}};
  for (std::size_t i = 0; i < layout.shards.size(); ++i) {
    const auto& shard = layout.shards[i];
    std::vector<CopyRange> ranges;
    ranges.reserve(shard.tensor_indices.size());
    for (const auto ordinal : shard.tensor_indices) ranges.push_back(tensors[ordinal].source);
    auto copied = CopyIdentityShard(staging_directory_fd, shard.layout.member_name,
                                   shard.layout.header_prefix, ranges);
    if (!copied.ok()) return copied.status();
    if (copied->file_bytes != shard.layout.file_bytes)
      return Status::Internal("copied shard byte ledger differs; staging members retained");
    manifest.shards.push_back({shard.name_space, shard.layout.member_name,
        copied->file_bytes, static_cast<std::uint32_t>(ranges.size()),
        copied->file_bytes - shard.layout.header_prefix.size(), copied->file_sha256,
        metadata->bound_layout.shard_layout_roots[i], {}});
  }
  for (const auto& [fd, initial] : sources) {
    struct stat observed{};
    if (::fstat(fd, &observed) != 0 || !SameSource(initial, observed))
      return Status::FailedPrecondition("generation source changed during copy; staging members retained");
  }
  auto encoded = DeepSeekRuntimeArtifactManifest::Encode(manifest);
  if (!encoded.ok()) return encoded.status();
  return CopiedGeneration{std::move(*metadata), std::move(*encoded)};
}
Status WriteGenerationMetadata(int staging_directory_fd, const CopiedGeneration& generation) {
  auto manifest = DeepSeekRuntimeArtifactManifest::Parse(
      generation.manifest.json, generation.manifest.artifact_root);
  if (!manifest.ok()) return manifest.status();
  auto geometry = manifest->validate_flash_0731_geometry();
  if (!geometry.ok()) return geometry;
  if (manifest->world_size() != 1 || manifest->dspark_enabled())
    return Status::InvalidArgument("generation metadata writer requires PP1 without DSpark");
  auto pipeline = DeepSeekPipelinePlan::Create(1, false);
  if (!pipeline.ok()) return pipeline.status();
  const auto& records_json = generation.metadata.runtime_records.json;
  auto records = DeepSeekRuntimeRecordsManifest::Parse(
      records_json, manifest->runtime_records(), *pipeline);
  if (!records.ok()) return records.status();
  const auto& index_json = generation.metadata.bound_layout.layout.index_json;
  auto index = SafetensorsShardIndex::Parse(index_json);
  if (!index.ok()) return index.status();
  if (index_json.size() != manifest->index().file_bytes ||
      index->total_size() != manifest->tensor_bytes() ||
      index->bindings().size() != records->records().size())
    return Status::InvalidArgument("generation index and record ledgers differ");
  for (std::size_t i = 0; i < index->bindings().size(); ++i) {
    if (index->bindings()[i].tensor_name != records->records()[i].tensor_name ||
        index->bindings()[i].shard_name != records->records()[i].shard_name)
      return Status::InvalidArgument("generation index and record name joins differ");
  }
  // Validate all in-memory hashes before creating any metadata member.
  auto index_hash = sha256(std::as_bytes(std::span(index_json)));
  auto manifest_hash = sha256(std::as_bytes(std::span(generation.manifest.json)));
  if (!index_hash.ok()) return index_hash.status();
  if (!manifest_hash.ok()) return manifest_hash.status();
  if (*index_hash != manifest->index().object_sha256 ||
      *manifest_hash != generation.manifest.object_sha256)
    return Status::InvalidArgument("generation metadata object hash differs before write");
  auto written = WriteMetadataMember(staging_directory_fd, "model.safetensors.index.json",
                                    index_json, *index_hash);
  if (!written.ok()) return written.status();
  written = WriteMetadataMember(staging_directory_fd, "pih.runtime-records.json",
                                records_json, manifest->runtime_records().object_sha256);
  if (!written.ok()) return written.status();
  written = WriteMetadataMember(staging_directory_fd, "pih.manifest.json",
                                generation.manifest.json, *manifest_hash);
  if (!written.ok()) return written.status();
  return Status::Ok();
}
}  // namespace pih::offline_deepseek
