#pragma once
#include "weight_catalog.h"
#include "weight_partition_copy.h"

namespace pih::deepseek_v41 {
struct OutputWeightShard final {
  std::string name;
  std::uint64_t file_bytes = 0;
  std::vector<std::byte> prefix;
};
// Offline safetensors layout derived solely from the admitted runtime inventory.
// The returned catalog reparses all generated headers through the runtime reader.
// Payload bytes and source provenance are not validated or published here.
class BackboneWeightOutputLayout final {
 public:
  static Result<BackboneWeightOutputLayout> Create(const FlashConfig& config,
      std::uint32_t world, std::uint32_t rank, std::uint64_t device_budget);
  std::span<const OutputWeightShard> shards() const noexcept { return shards_; }
  const BackboneWeightCatalog& catalog() const noexcept { return catalog_; }
 private:
  BackboneWeightOutputLayout(std::vector<OutputWeightShard> shards,
                            BackboneWeightCatalog catalog)
      : shards_(std::move(shards)), catalog_(std::move(catalog)) {}
  std::vector<OutputWeightShard> shards_;
  BackboneWeightCatalog catalog_;
};

using WeightShardWriter = std::function<Status(
    std::string_view, std::uint64_t, std::span<const std::byte>)>;
// Emits complete safetensors headers and partitioned payloads at exact file
// offsets. The writer owns unpublished files and must complete each span.
// Success returns geometry, NOT a digest/authenticity/publication receipt.
Result<BackboneWeightOutputLayout> WriteCanonicalBackboneRank(
    const FlashConfig& config, std::uint32_t world, std::uint32_t rank,
    std::uint64_t device_budget, std::span<const RuntimeWeight> source_inventory,
    const CanonicalWeightReader& read, const WeightShardWriter& write,
    std::span<std::byte> workspace);
}  // namespace pih::deepseek_v41
