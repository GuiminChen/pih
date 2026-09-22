#pragma once
#include "weight_files.h"
#include "weight_source_checkpoint.h"

namespace pih::deepseek_v41 {
struct MaterializedWeightRank final {
  Sha256Digest manifest_sha256;
  // Nonzero only for source-checkpoint conversion with provenance metadata.
  Sha256Digest provenance_sha256;
  std::uint64_t file_bytes = 0;
  std::uint32_t shard_count = 0;
};
// Linux offline operation. The borrowed descriptor must name a private empty
// directory owned by this process's effective user. Creates files exclusively,
// verifies readback, makes members read-only and syncs them plus the directory.
// Does not rename/publish the directory; failure retains partial files, never
// returns an admission receipt and never deletes caller data. The caller must
// exclude concurrent same-user/root writers and keep the source immutable.
// Source must already be an authenticated canonical TP1 rank, not raw HF data.
Result<MaterializedWeightRank> MaterializeCanonicalWeightRank(
    int staging_directory_fd, const BackboneWeightFiles& source,
    const FlashConfig& config, std::uint32_t world, std::uint32_t rank,
    std::uint64_t device_budget);
// Converts the admitted HF source into a canonical TP1 rank in private staging.
// Saves the exact exclusions and source expectation digest in
// conversion.provenance.json. Same failure/no-publication rules as above.
Result<MaterializedWeightRank> MaterializeSourceBackbone(
    int staging_directory_fd, const WeightSourceCheckpoint& source,
    std::span<const std::string> excluded_names, std::uint64_t device_budget);
// Add worker metadata to still-private, writable staging (not a published rank).
// Caller binds expected_config to the rank manifest. No overwrite/publication.
Result<std::uint64_t> MaterializeRuntimeMetadata(int staging_directory_fd,
    std::string_view config_json, const Sha256Digest& expected_config,
    std::span<const std::byte> map, const Sha256Digest& expected_map);
}  // namespace pih::deepseek_v41
