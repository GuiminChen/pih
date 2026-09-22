#include "weight_catalog.h"
#include "pih/model/safetensors_header.h"
#include <algorithm>
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
bool IsWeightShardMember(std::string_view name) noexcept {
  if (name.empty() || name.size() > 128 || !name.ends_with(".safetensors") ||
      name.front() == '.' || name.find("..") != std::string_view::npos) return false;
  for (const unsigned char c : name)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) return false;
  return true;
}
namespace {
DType Storage(WeightStorage storage) {
  switch (storage) {
    case WeightStorage::kBF16: return DType::kBFloat16;
    case WeightStorage::kF32: return DType::kFloat32;
    case WeightStorage::kE4M3FN: return DType::kFloat8E4M3;
    case WeightStorage::kE8M0: return DType::kFloat8E8M0;
    // Canonical byte container, not signed numeric int8 or logical FP4 shape.
    case WeightStorage::kPackedE2M1: return DType::kUInt8;
  }
  return DType::kBool;
}
}
Result<BackboneWeightCatalog> BackboneWeightCatalog::Create(const FlashConfig& config,
    std::uint32_t world, std::uint32_t rank,
    std::span<const WeightShardPrefix> shards, std::uint64_t budget) {
  if (shards.empty() || shards.size() > kMaximumShards || !budget)
    return Status::InvalidArgument("V4.1 runtime shard count or device budget invalid");
  std::uint64_t header_bytes = 0;
  for (const auto& shard : shards) {
    if (!IsWeightShardMember(shard.name) || shard.prefix.size() > kMaximumHeaderBytes - header_bytes)
      return Status::InvalidArgument("V4.1 runtime shard name or header budget invalid");
    header_bytes += shard.prefix.size();
  }
  auto inventory = BackboneWeightInventory::Create(config, world, rank);
  if (!inventory.ok()) return inventory.status();
  if (inventory->bytes() > budget)
    return Status::ResourceExhausted("V4.1 runtime weights exceed device payload budget");
  try {
    BackboneWeightCatalog result;
    result.world_ = world; result.rank_ = rank; result.config_sha256_ = config.config_sha256();
    result.weights_.reserve(inventory->weights().size());
    result.shards_.reserve(shards.size());
    for (const auto& weight : inventory->weights()) {
      const auto size = weight.tensor.bytes;
      if (size > std::numeric_limits<std::uint64_t>::max() - 255)
        return Status::ResourceExhausted("V4.1 weight alignment overflow");
      const auto padded = (size + 255) & ~std::uint64_t{255};
      if (padded > budget - result.device_bytes_)
        return Status::ResourceExhausted("V4.1 aligned weights exceed device budget");
      result.weights_.push_back({weight, UINT32_MAX, 0, result.device_bytes_});
      result.device_bytes_ += padded;
    }
    for (std::size_t shard_index = 0; shard_index < shards.size(); ++shard_index) {
      const auto& shard = shards[shard_index];
      for (const auto& admitted : result.shards_)
        if (admitted.name == shard.name)
          return Status::InvalidArgument("Duplicate V4.1 runtime shard name");
      auto header = SafetensorsHeader::ParsePrefix(shard.prefix, shard.file_bytes);
      if (!header.ok()) return header.status();
      // No hidden prefix payload or multiple interpretations of the input.
      if (shard.prefix.size() != 8 + header->header_bytes())
        return Status::InvalidArgument("V4.1 shard prefix must contain exactly its header");
      result.shards_.push_back({std::string(shard.name), shard.file_bytes});
      for (const auto& tensor : header->tensors()) {
        auto it = std::lower_bound(result.weights_.begin(), result.weights_.end(), tensor.name,
            [](const auto& a, const auto& name) { return a.weight.tensor.name < name; });
        if (it == result.weights_.end() || it->weight.tensor.name != tensor.name || it->shard != UINT32_MAX)
          return Status::InvalidArgument("Unknown or duplicate V4.1 runtime tensor across shards");
        const auto& expected = it->weight.tensor;
        if (tensor.dtype != Storage(expected.storage) || tensor.shape.size() != expected.dimensions ||
            tensor.shape[0] != expected.rows ||
            (expected.dimensions == 2 && tensor.shape[1] != expected.columns) ||
            tensor.file_end - tensor.file_begin != expected.bytes)
          return Status::InvalidArgument("V4.1 runtime shard tensor storage or geometry differs");
        it->shard = static_cast<std::uint32_t>(shard_index);
        it->file_offset = tensor.file_begin;
      }
    }
    for (const auto& weight : result.weights_)
      if (weight.shard == UINT32_MAX)
        return Status::InvalidArgument("Missing V4.1 backbone runtime tensor");
    return result;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 runtime catalog allocation failed");
  }
}
const LocatedWeight* BackboneWeightCatalog::Find(std::string_view name) const noexcept {
  const auto it = std::lower_bound(weights_.begin(), weights_.end(), name,
      [](const auto& a, auto key) { return a.weight.tensor.name < key; });
  return it == weights_.end() || it->weight.tensor.name != name ? nullptr : &*it;
}
}  // namespace pih::deepseek_v41
