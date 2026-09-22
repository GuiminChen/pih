#pragma once
#include "engram_weights.h"
#include <string>
#include <vector>

namespace pih::deepseek_v41 {
enum class WeightStorage { kBF16, kF32, kE4M3FN, kE8M0, kPackedE2M1 };
// Physical on-disk dimensions, not logical FP4 dimensions. Vectors use
// dimensions=1 and columns=1. This describes canonical runtime artifacts,
// never raw HF tensors (which may require conversion before admission).
struct RuntimeWeight final {
  std::string name;
  WeightStorage storage{};
  std::uint32_t dimensions = 2;
  std::uint64_t rows = 0, columns = 0, bytes = 0;
};
struct WeightPartition final {
  RuntimeWeight tensor;
  // -1: replicated, or a whole expert assigned by its global ID in name.
  // 0/1: contiguous partition along the physical row/column dimension.
  int axis = -1;
  std::uint64_t first = 0, valid = 0, padding = 0;
};
class BackboneWeightInventory final {
 public:
  // Text backbone only: 40 layers, no vision/MTP/DSpark members. Those must be
  // explicitly separated by a future authenticated conversion manifest.
  // Norms and Engram q/k use canonical BF16; router/head are canonical F32.
  static Result<BackboneWeightInventory> Create(const FlashConfig& config,
      std::uint32_t world_size, std::uint32_t rank);
  std::span<const WeightPartition> weights() const noexcept { return weights_; }
  std::uint64_t bytes() const noexcept { return bytes_; }
  // Exact set, order-independent. Reject missing, unknown, duplicate, wrong
  // dtype/dimensions/extent members. Metadata only: no authenticity claim.
  Status Validate(std::span<const RuntimeWeight> supplied) const;
 private:
  BackboneWeightInventory() = default;
  std::vector<WeightPartition> weights_;
  std::uint64_t bytes_ = 0;
};
}  // namespace pih::deepseek_v41
