#pragma once

#include "weight_inventory.h"
#include <functional>

namespace pih::deepseek_v41 {
struct WeightCopyRange final {
  std::uint64_t source_offset = 0;
  std::uint64_t destination_offset = 0;
  std::uint64_t bytes = 0;
  bool padding = false;
  std::byte fill_byte{};
};

// Offline conversion stage for already-canonical full tensors, NOT raw HF
// dtype conversion or authenticity admission. No file offsets are serialized.
class WeightPartitionCopy final {
 public:
  static Result<WeightPartitionCopy> Create(
      const RuntimeWeight& full, const WeightPartition& partition);
  // One contiguous source/padding range, bounded by caller workspace. Iterating
  // by destination offset avoids allocating a plan proportional to tensor rows.
  Result<WeightCopyRange> Next(std::uint64_t destination_offset,
                              std::uint64_t maximum_bytes) const;
  using Reader = std::function<Status(std::uint64_t, std::span<std::byte>)>;
  using Writer = std::function<Status(std::uint64_t, std::span<const std::byte>)>;
  // Reader/writer must complete each exact span or return failure. Any failure
  // invalidates the partially written destination; no publication is performed.
  Status Copy(const Reader& read, const Writer& write, std::span<std::byte> workspace) const;
  std::uint64_t bytes() const noexcept { return destination_bytes_; }
 private:
  WeightPartitionCopy() = default;
  std::uint64_t source_bytes_ = 0, destination_bytes_ = 0;
  std::uint64_t source_row_bytes_ = 0, destination_row_bytes_ = 0;
  std::uint64_t source_first_ = 0, valid_bytes_ = 0;
  int axis_ = -1;
  std::byte padding_byte_{};
};

// Copy one rank from a complete, canonical text-backbone source. Source metadata
// must exactly match the config-derived TP1 inventory; arbitrary caller-defined
// shard geometry is never accepted. File authentication and transactional output
// publication belong to the caller. Vision/MTP/raw checkpoint members are rejected.
using CanonicalWeightReader = std::function<Status(
    std::string_view, std::uint64_t, std::span<std::byte>)>;
using PartitionWeightWriter = std::function<Status(
    const RuntimeWeight&, std::uint64_t, std::span<const std::byte>)>;
Status CopyCanonicalBackboneRank(const FlashConfig& config,
    std::uint32_t world, std::uint32_t rank,
    std::span<const RuntimeWeight> source_inventory,
    const CanonicalWeightReader& read, const PartitionWeightWriter& write,
    std::span<std::byte> workspace);
}  // namespace pih::deepseek_v41
