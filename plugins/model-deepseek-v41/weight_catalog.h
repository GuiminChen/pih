#pragma once
#include "weight_inventory.h"

namespace pih::deepseek_v41 {
bool IsWeightShardMember(std::string_view name) noexcept;
struct WeightShardPrefix final {
  // Flat artifact member name, not a path. Prefix includes the eight-byte
  // little-endian length and the entire safetensors JSON header.
  std::string_view name;
  std::uint64_t file_bytes = 0;
  std::span<const std::byte> prefix;
};
struct WeightShard final {
  std::string name;
  std::uint64_t file_bytes = 0;
};
struct LocatedWeight final {
  WeightPartition weight;
  std::uint32_t shard = 0;
  std::uint64_t file_offset = 0, device_offset = 0;
};
// Owned, immutable metadata joining actual runtime shard headers to the exact
// backbone inventory. Not a source converter or payload authentication proof.
class BackboneWeightCatalog final {
 public:
  static constexpr std::size_t kMaximumShards = 256;
  static constexpr std::uint64_t kMaximumHeaderBytes = 256ULL * 1024 * 1024;
  static Result<BackboneWeightCatalog> Create(const FlashConfig& config,
      std::uint32_t world, std::uint32_t rank,
      std::span<const WeightShardPrefix> shards, std::uint64_t device_budget);
  std::span<const LocatedWeight> weights() const noexcept { return weights_; }
  std::span<const WeightShard> shards() const noexcept { return shards_; }
  const LocatedWeight* Find(std::string_view name) const noexcept;
  // All tensor starts and the total allocation extent are 256-byte aligned.
  // Scratch, caches, and allocator overhead are not included.
  std::uint64_t device_bytes() const noexcept { return device_bytes_; }
  std::uint32_t world_size() const noexcept { return world_; }
  std::uint32_t rank() const noexcept { return rank_; }
  const Sha256Digest& config_sha256() const noexcept { return config_sha256_; }
 private:
  BackboneWeightCatalog() = default;
  std::vector<LocatedWeight> weights_;
  std::vector<WeightShard> shards_;
  std::uint64_t device_bytes_ = 0;
  std::uint32_t world_ = 0, rank_ = 0;
  Sha256Digest config_sha256_;
};
}  // namespace pih::deepseek_v41
