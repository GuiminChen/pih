#pragma once
#include "shard_layout.h"

namespace pih::offline_deepseek {
struct GenerationShardLayout final {
  std::string name_space;
  ShardLayout layout;
  // Indices into the strictly name-sorted admitted input array.
  std::vector<std::size_t> tensor_indices;
};
struct GenerationLayout final {
  std::vector<GenerationShardLayout> shards;
  std::string index_json;
  Sha256Digest index_sha256;
  std::uint64_t tensor_bytes = 0;
  std::uint64_t total_shard_bytes = 0;
};
// PP1, DSpark disabled. Produces canonical byte layout/index only, not the
// disposition/layout authority roots or evidence of official source provenance.
Result<GenerationLayout> BuildGenerationLayout(std::span<const IdentityTensor> tensors);

struct GenerationLayoutAuthority final {
  Sha256Digest source_inventory_root;
  Sha256Digest source_payload_closure_root;
  Sha256Digest disposition_root;
  // Exactly one owner in the PP1 / DSpark-disabled projection.
  Sha256Digest owner_root;
};
struct BoundGenerationLayout final {
  GenerationLayout layout;
  // Tensor roots follow the input's strict name order; shard roots follow
  // endpoint, layers.0, ..., layers.42 (not lexicographic namespace order).
  std::vector<Sha256Digest> layout_record_roots;
  std::vector<Sha256Digest> shard_layout_roots;
  Sha256Digest weight_map_root;
  Sha256Digest index_root;
  Sha256Digest owner_projection_root;
  Sha256Digest logical_layout_root;
  Sha256Digest layout_root;
  std::uint64_t total_header_bytes = 0;
  std::uint64_t total_artifact_bytes = 0;
};
// Rebuilds and binds the complete fixed PP1 geometry. Supplied authority roots
// must already be admitted by the caller; hashing them does not establish
// provenance, payload integrity, disposition correctness or immutable storage.
Result<BoundGenerationLayout> BindGenerationLayout(
    std::span<const IdentityTensor> tensors,
    std::span<const TensorLayoutAuthority> tensor_authorities,
    const GenerationLayoutAuthority& authority);
}  // namespace pih::offline_deepseek
