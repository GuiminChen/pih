#include "weight_output_layout.h"
#include "pih/core/canonical_json.h"
#include "pih/model/safetensors_header.h"
#include <array>
#include <algorithm>
#include <charconv>
#include <new>

namespace pih::deepseek_v41 {
namespace {
Result<std::string> StorageName(WeightStorage storage) {
  switch (storage) {
    case WeightStorage::kBF16: return std::string("BF16");
    case WeightStorage::kF32: return std::string("F32");
    case WeightStorage::kE4M3FN: return std::string("F8_E4M3");
    case WeightStorage::kE8M0: return std::string("F8_E8M0");
    case WeightStorage::kPackedE2M1: return std::string("U8");
  }
  return Status::Internal("V4.1 output inventory has unknown storage");
}
Result<std::size_t> ShardIndex(std::string_view name) {
  if (name == "embed.weight" || name == "norm.weight" || name == "head.weight")
    return std::size_t{0};
  if (!name.starts_with("layers."))
    return Status::Internal("V4.1 output inventory namespace unknown");
  name.remove_prefix(7);
  const auto end = name.find('.');
  if (end == std::string_view::npos)
    return Status::Internal("V4.1 output inventory layer absent");
  unsigned layer = FlashConfig::kMainLayers;
  const auto parsed = std::from_chars(name.data(), name.data() + end, layer);
  if (parsed.ec != std::errc{} || parsed.ptr != name.data() + end ||
      layer >= FlashConfig::kMainLayers)
    return Status::Internal("V4.1 output inventory layer invalid");
  return static_cast<std::size_t>(layer + 1);
}
}
Result<BackboneWeightOutputLayout> BackboneWeightOutputLayout::Create(
    const FlashConfig& config, std::uint32_t world, std::uint32_t rank,
    std::uint64_t device_budget) {
  auto inventory = BackboneWeightInventory::Create(config, world, rank);
  if (!inventory.ok()) return inventory.status();
  if (!device_budget || inventory->bytes() > device_budget)
    return Status::ResourceExhausted("V4.1 output layout exceeds device payload budget");
  try {
    constexpr std::size_t count = FlashConfig::kMainLayers + 1;
    constexpr std::uint64_t maximum = 512ULL << 30;
    std::array<JsonValue::Object, count> headers;
    std::array<std::uint64_t, count> cursors{};
    for (const auto& partition : inventory->weights()) {
      const auto& tensor = partition.tensor;
      auto slot = ShardIndex(tensor.name);
      if (!slot.ok()) return slot.status();
      auto& cursor = cursors[*slot];
      if (tensor.bytes > maximum - cursor)
        return Status::ResourceExhausted("V4.1 output shard payload exceeds 512 GiB");
      auto storage = StorageName(tensor.storage);
      if (!storage.ok()) return storage.status();
      JsonValue::Array shape{JsonValue(static_cast<std::int64_t>(tensor.rows))};
      if (tensor.dimensions == 2) shape.emplace_back(static_cast<std::int64_t>(tensor.columns));
      const auto end = cursor + tensor.bytes;
      headers[*slot].emplace_back(tensor.name, JsonValue(JsonValue::Object{
          {"dtype", JsonValue(std::move(*storage))}, {"shape", JsonValue(std::move(shape))},
          {"data_offsets", JsonValue(JsonValue::Array{
              JsonValue(static_cast<std::int64_t>(cursor)), JsonValue(static_cast<std::int64_t>(end))})}}));
      cursor = end;
    }
    std::vector<OutputWeightShard> shards;
    shards.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      if (headers[index].empty()) return Status::Internal("V4.1 output shard is empty");
      auto json = canonical_ascii_json(JsonValue(std::move(headers[index])), SafetensorsHeader::kMaxHeaderBytes);
      if (!json.ok()) return json.status();
      const auto padded = (json->size() + 7) & ~std::size_t{7};
      if (padded > SafetensorsHeader::kMaxHeaderBytes || cursors[index] > maximum - padded - 8)
        return Status::ResourceExhausted("V4.1 output header or file exceeds budget");
      OutputWeightShard shard;
      shard.name = index == 0 ? "model-endpoint.safetensors" :
          "model-layers-" + std::to_string(index - 1) + ".safetensors";
      shard.prefix.resize(padded + 8, std::byte{' '});
      for (unsigned byte = 0; byte < 8; ++byte)
        shard.prefix[byte] = static_cast<std::byte>((static_cast<std::uint64_t>(padded) >> (8 * byte)) & 255);
      for (std::size_t byte = 0; byte < json->size(); ++byte)
        shard.prefix[8 + byte] = static_cast<std::byte>((*json)[byte]);
      shard.file_bytes = shard.prefix.size() + cursors[index];
      shards.push_back(std::move(shard));
    }
    std::vector<WeightShardPrefix> prefixes;
    prefixes.reserve(shards.size());
    for (const auto& shard : shards) prefixes.push_back({shard.name, shard.file_bytes, shard.prefix});
    auto catalog = BackboneWeightCatalog::Create(config, world, rank, prefixes, device_budget);
    if (!catalog.ok()) return catalog.status();
    return BackboneWeightOutputLayout(std::move(shards), std::move(*catalog));
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 output layout allocation failed");
  }
}
Result<BackboneWeightOutputLayout> WriteCanonicalBackboneRank(
    const FlashConfig& config, std::uint32_t world, std::uint32_t rank,
    std::uint64_t device_budget, std::span<const RuntimeWeight> source_inventory,
    const CanonicalWeightReader& read, const WeightShardWriter& write,
    std::span<std::byte> workspace) {
  if (!read || !write || workspace.empty() || workspace.size() > 1024 * 1024)
    return Status::InvalidArgument("V4.1 output requires bounded workspace and exact I/O callbacks");
  auto full = BackboneWeightInventory::Create(config, 1, 0);
  if (!full.ok()) return full.status();
  auto admitted = full->Validate(source_inventory);
  if (!admitted.ok()) return admitted;
  auto layout = BackboneWeightOutputLayout::Create(config, world, rank, device_budget);
  if (!layout.ok()) return layout.status();
  try {
    for (const auto& shard : layout->shards()) {
      for (std::size_t offset = 0; offset < shard.prefix.size();) {
        const auto count = std::min(workspace.size(), shard.prefix.size() - offset);
        auto status = write(shard.name, offset,
            std::span<const std::byte>(shard.prefix).subspan(offset, count));
        if (!status.ok()) return status;
        offset += count;
      }
    }
    auto copied = CopyCanonicalBackboneRank(config, world, rank, source_inventory, read,
        [&](const RuntimeWeight& tensor, std::uint64_t offset, std::span<const std::byte> bytes) {
          const auto* location = layout->catalog().Find(tensor.name);
          if (!location || location->shard >= layout->shards().size() ||
              offset > location->weight.tensor.bytes ||
              bytes.size() > location->weight.tensor.bytes - offset)
            return Status::Internal("V4.1 output copy range differs from runtime catalog");
          return write(layout->shards()[location->shard].name, location->file_offset + offset, bytes);
        }, workspace);
    if (!copied.ok()) return copied;
    return std::move(*layout);
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 output allocation failed; discard unpublished files");
  } catch (...) {
    return Status::Internal("V4.1 output callback failed; discard unpublished files");
  }
}
}  // namespace pih::deepseek_v41
