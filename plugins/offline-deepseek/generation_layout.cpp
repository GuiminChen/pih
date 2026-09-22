#include "generation_layout.h"
#include "tensor_inventory.h"
#include <charconv>
#include "pih/core/canonical_hash.h"
#include "pih/core/canonical_json.h"
#include "pih/model/safetensors_shard_index.h"

namespace pih::offline_deepseek {
Result<GenerationLayout> BuildGenerationLayout(std::span<const IdentityTensor> tensors) {
  constexpr std::size_t tensor_count = 67'612;
  constexpr std::uint64_t tensor_bytes = 156'015'698'140ULL;
  if (tensors.size() != tensor_count)
    return Status::InvalidArgument("PP1 tensor count differs from fixed 0731 geometry");
  auto expected_names = ExpectedTensorNames(false);
  if (!expected_names.ok()) return expected_names.status();
  for (std::size_t i = 0; i < tensors.size(); ++i)
    if (tensors[i].name != (*expected_names)[i])
      return Status::InvalidArgument("PP1 tensor name differs from frozen 0731 inventory");
  for (const auto& tensor : tensors) {
    auto status = ValidateRoutedExpertGeometry(tensor.name, tensor.dtype, tensor.shape);
    if (!status.ok()) return status;
    status = ValidatePp1DenseGeometry(tensor.name, tensor.dtype, tensor.shape);
    if (!status.ok()) return status;
  }
  GenerationLayout result;
  result.shards.resize(44);
  result.shards[0].name_space = "endpoint";
  for (unsigned layer = 0; layer < 43; ++layer)
    result.shards[layer + 1].name_space = "layers." + std::to_string(layer);
  std::string_view previous;
  for (std::size_t ordinal = 0; ordinal < tensors.size(); ++ordinal) {
    const auto& tensor = tensors[ordinal];
    if (tensor.name.empty() || tensor.name <= previous || tensor.name.size() > 512 ||
        tensor.shape.empty() || tensor.shape.size() > 8 ||
        tensor.source.bytes == 0 || tensor.source.bytes > tensor_bytes - result.tensor_bytes)
      return Status::InvalidArgument("PP1 tensor names or payload ledger invalid");
    previous = tensor.name;
    std::size_t group = 0;
    if (tensor.name.starts_with("layers.")) {
      const auto dot = tensor.name.find('.', 7);
      if (dot == tensor.name.npos || dot == 7)
        return Status::InvalidArgument("PP1 layer namespace is malformed");
      const auto number = std::string_view(tensor.name).substr(7, dot - 7);
      unsigned layer = 43;
      const auto parsed = std::from_chars(number.data(), number.data() + number.size(), layer);
      if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() ||
          layer >= 43 || (number.size() > 1 && number.front() == '0'))
        return Status::InvalidArgument("PP1 layer namespace is outside fixed geometry");
      group = layer + 1;
    }
    if (result.shards[group].tensor_indices.size() >= 4096)
      return Status::InvalidArgument("PP1 shard tensor count exceeds its bound");
    result.shards[group].tensor_indices.push_back(ordinal);
    result.tensor_bytes += tensor.source.bytes;
  }
  if (result.tensor_bytes != tensor_bytes)
    return Status::InvalidArgument("PP1 payload bytes differ from fixed 0731 geometry");
  JsonValue::Object weight_map;
  weight_map.reserve(tensors.size());
  for (auto& shard : result.shards) {
    std::vector<IdentityTensor> selected;
    selected.reserve(shard.tensor_indices.size());
    for (const auto ordinal : shard.tensor_indices) selected.push_back(tensors[ordinal]);
    auto layout = BuildShardLayout(shard.name_space, selected);
    if (!layout.ok()) return layout.status();
    shard.layout = std::move(*layout);
    result.total_shard_bytes += shard.layout.file_bytes;
    for (const auto ordinal : shard.tensor_indices)
      weight_map.emplace_back(tensors[ordinal].name, JsonValue(shard.layout.member_name));
  }
  auto index = canonical_ascii_json(JsonValue(JsonValue::Object{
      {"metadata", JsonValue(JsonValue::Object{
          {"total_size", JsonValue(static_cast<std::int64_t>(result.tensor_bytes))}})},
      {"weight_map", JsonValue(std::move(weight_map))}}), SafetensorsShardIndex::kMaximumIndexBytes);
  if (!index.ok()) return index.status();
  auto parsed_index = SafetensorsShardIndex::Parse(*index);
  if (!parsed_index.ok()) return parsed_index.status();
  if (parsed_index->bindings().size() != tensors.size() ||
      parsed_index->shard_names().size() != result.shards.size() ||
      parsed_index->total_size() != result.tensor_bytes)
    return Status::Internal("native PP1 generated index does not close its layout");
  auto digest = sha256(std::as_bytes(std::span(*index)));
  if (!digest.ok()) return digest.status();
  result.index_sha256 = *digest;
  result.index_json = std::move(*index);
  return result;
}
namespace {
Status HashText(CanonicalHashBuilder& hash, std::uint16_t id, std::string_view value) {
  return hash.add_bytes(id, std::as_bytes(std::span(value)));
}
Result<Sha256Digest> HashSet(std::string_view domain, std::span<const Sha256Digest> roots) {
  if (roots.empty() || roots.size() > 44)
    return Status::InvalidArgument("generation layout root count invalid");
  auto hash = CanonicalHashBuilder::Create(domain, static_cast<std::uint32_t>(roots.size() + 1));
  if (!hash.ok()) return hash.status();
  auto status = hash->add_u32(1, static_cast<std::uint32_t>(roots.size()));
  for (std::size_t i = 0; status.ok() && i < roots.size(); ++i)
    status = hash->add_hash(static_cast<std::uint16_t>(100 + i), roots[i]);
  if (!status.ok()) return status;
  return hash->finalize();
}
constexpr std::string_view kLayoutAbi = "deepseek_v4_flash_0731_target_layout_v1";
constexpr std::string_view kFamily = "deepseek_v4_flash_0731";
}  // namespace

Result<BoundGenerationLayout> BindGenerationLayout(
    std::span<const IdentityTensor> tensors,
    std::span<const TensorLayoutAuthority> tensor_authorities,
    const GenerationLayoutAuthority& authority) {
  if (tensor_authorities.size() != tensors.size() ||
      authority.source_inventory_root == Sha256Digest{} ||
      authority.source_payload_closure_root == Sha256Digest{} ||
      authority.disposition_root == Sha256Digest{} || authority.owner_root == Sha256Digest{})
    return Status::InvalidArgument("generation layout authority missing or cardinality invalid");
  auto layout = BuildGenerationLayout(tensors);
  if (!layout.ok()) return layout.status();
  BoundGenerationLayout result;
  result.layout = std::move(*layout);
  result.layout_record_roots.resize(tensors.size());
  std::vector<Sha256Digest> weight_roots;
  for (const auto& shard : result.layout.shards) {
    std::vector<IdentityTensor> selected;
    std::vector<TensorLayoutAuthority> selected_authorities;
    selected.reserve(shard.tensor_indices.size());
    selected_authorities.reserve(shard.tensor_indices.size());
    for (const auto ordinal : shard.tensor_indices) {
      selected.push_back(tensors[ordinal]);
      selected_authorities.push_back(tensor_authorities[ordinal]);
    }
    auto bound = BindShardLayout(shard.name_space, selected, selected_authorities);
    if (!bound.ok()) return bound.status();
    // Both builders derive bytes from the same input. Never bind an authority
    // to a caller-provided or divergent mutable byte layout.
    if (bound->layout.member_name != shard.layout.member_name ||
        bound->layout.header_prefix != shard.layout.header_prefix ||
        bound->layout.file_ranges != shard.layout.file_ranges ||
        bound->layout.file_bytes != shard.layout.file_bytes)
      return Status::Internal("generation and bound shard layouts disagree");
    for (std::size_t i = 0; i < shard.tensor_indices.size(); ++i)
      result.layout_record_roots[shard.tensor_indices[i]] = bound->layout_record_roots[i];
    result.shard_layout_roots.push_back(bound->shard_layout_root);
    weight_roots.push_back(bound->weight_map_set_root);
    result.total_header_bytes += shard.layout.header_prefix.size() - 8;
  }
  result.total_artifact_bytes = result.layout.total_shard_bytes + result.layout.index_json.size();
  auto weights = HashSet("pih:deepseek-v4-flash-0731-target-weight-map:v1", weight_roots);
  if (!weights.ok()) return weights.status();
  result.weight_map_root = *weights;
  const auto count = static_cast<std::uint32_t>(tensors.size());
  const auto shards = static_cast<std::uint32_t>(result.layout.shards.size());
  auto hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-index:v1", 6);
  if (!hash.ok()) return hash.status();
  auto status = hash->add_u64(1, result.layout.index_json.size());
  if (status.ok()) status = hash->add_hash(2, result.layout.index_sha256);
  if (status.ok()) status = hash->add_hash(3, result.weight_map_root);
  if (status.ok()) status = hash->add_u32(4, count);
  if (status.ok()) status = hash->add_u64(5, result.layout.tensor_bytes);
  if (status.ok()) status = hash->add_u32(6, shards);
  if (!status.ok()) return status;
  auto index = hash->finalize();
  if (!index.ok()) return index.status();
  result.index_root = *index;
  hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-logical-layout:v1", 13 + shards);
  if (!hash.ok()) return hash.status();
  status = HashText(*hash, 1, kLayoutAbi);
  if (status.ok()) status = HashText(*hash, 2, kFamily);
  if (status.ok()) status = hash->add_hash(3, authority.source_inventory_root);
  if (status.ok()) status = hash->add_hash(4, authority.source_payload_closure_root);
  if (status.ok()) status = hash->add_u32(5, 0);  // DSpark disabled.
  if (status.ok()) status = hash->add_u32(6, count);
  if (status.ok()) status = hash->add_u64(7, result.layout.tensor_bytes);
  if (status.ok()) status = hash->add_u32(8, shards);
  if (status.ok()) status = hash->add_u64(9, result.total_header_bytes);
  if (status.ok()) status = hash->add_u64(10, result.layout.total_shard_bytes);
  if (status.ok()) status = hash->add_u64(11, result.layout.index_json.size());
  if (status.ok()) status = hash->add_u64(12, result.total_artifact_bytes);
  if (status.ok()) status = hash->add_hash(13, result.index_root);
  for (std::size_t i = 0; status.ok() && i < result.shard_layout_roots.size(); ++i)
    status = hash->add_hash(static_cast<std::uint16_t>(100 + i), result.shard_layout_roots[i]);
  if (!status.ok()) return status;
  auto logical = hash->finalize();
  if (!logical.ok()) return logical.status();
  result.logical_layout_root = *logical;
  auto owners = HashSet("pih:deepseek-v4-flash-0731-target-layout-owner-projection:v1",
                        std::span(&authority.owner_root, 1));
  if (!owners.ok()) return owners.status();
  result.owner_projection_root = *owners;
  hash = CanonicalHashBuilder::Create("pih:deepseek-v4-flash-0731-target-layout:v1", 12);
  if (!hash.ok()) return hash.status();
  status = HashText(*hash, 1, kLayoutAbi);
  if (status.ok()) status = HashText(*hash, 2, kFamily);
  if (status.ok()) status = hash->add_hash(3, authority.disposition_root);
  if (status.ok()) status = hash->add_hash(4, result.logical_layout_root);
  if (status.ok()) status = hash->add_u32(5, 0);
  if (status.ok()) status = hash->add_u32(6, 1);  // PP1.
  if (status.ok()) status = hash->add_hash(7, result.owner_projection_root);
  if (status.ok()) status = hash->add_u32(8, count);
  if (status.ok()) status = hash->add_u64(9, result.layout.tensor_bytes);
  if (status.ok()) status = hash->add_u32(10, shards);
  if (status.ok()) status = HashText(*hash, 11, "canonical_target_safetensors_layout_non_authorizing");
  if (status.ok()) status = HashText(*hash, 12, "hardware_evidence_open");
  if (!status.ok()) return status;
  auto root = hash->finalize();
  if (!root.ok()) return root.status();
  result.layout_root = *root;
  return result;
}
}  // namespace pih::offline_deepseek
