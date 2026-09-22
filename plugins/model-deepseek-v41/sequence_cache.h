#pragma once
#include "engram_launch.h"
#include "pih/contracts/nvidia_cuda_memory_v1.h"

namespace pih::deepseek_v41 {
struct CacheSegment final { std::uint64_t offset = 0, bytes = 0; };
struct LayerCacheLayout final {
  CacheSegment window, compressed, index_keys, pool_values, pool_scores;
  std::uint32_t compressed_capacity = 0;
};
struct LayerCacheRegions final {
  EngramDeviceRegion window, compressed, index_keys, pool_values, pool_scores;
  std::uint32_t compressed_capacity = 0;
};
// Single-sequence persistent cache arena, replicated on every TP rank. No
// per-step scores, selection masks, residuals, weights or expert workspace.
class SequenceCachePlan final {
 public:
  static Result<SequenceCachePlan> Create(const FlashConfig& config,
      std::uint32_t maximum_positions, std::uint64_t byte_budget);
  std::uint64_t bytes() const noexcept { return bytes_; }
  std::uint32_t maximum_positions() const noexcept { return maximum_positions_; }
  const Sha256Digest& config_sha256() const noexcept { return config_sha256_; }
  const std::array<LayerCacheLayout, 40>& layers() const noexcept { return layers_; }
  // Owner-only views. Nonowners have absent compressed/key/pool regions.
  Result<LayerCacheRegions> Bind(EngramDeviceRegion arena, std::uint32_t layer) const;
  Status ValidateAllocation(const pih_cuda_allocation_v1& allocation, std::int32_t device) const;
  // Preserves exact provider output in caller's empty ABI-initialized ledger,
  // even when status/ownership is ambiguous. No automatic free on failure.
  Status Allocate(const pih_nvidia_cuda_memory_api_v1& memory, std::int32_t device,
      pih_cuda_allocation_v1& output) const;
 private:
  SequenceCachePlan() = default;
  std::array<LayerCacheLayout, 40> layers_{};
  Sha256Digest config_sha256_;
  std::uint64_t bytes_ = 0;
  std::uint32_t maximum_positions_ = 0;
};
}  // namespace pih::deepseek_v41
