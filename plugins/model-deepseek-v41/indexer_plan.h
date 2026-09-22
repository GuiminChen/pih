#pragma once
#include "uploaded_bindings.h"
#include "sequence_cache.h"

namespace pih::deepseek_v41 {
class IndexerPlan final {
 public:
  static Result<IndexerPlan> Create(const FlashConfig& config, std::uint32_t tokens,
      std::uint32_t maximum_positions, std::uint32_t world, std::uint32_t rank, std::uint64_t budget);
  std::uint64_t bytes() const noexcept { return bytes_; }
  Result<IndexerPipelineLaunch> Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
      EngramDeviceRegion arena, std::uint32_t layer, std::uint32_t start, std::uint32_t tokens,
      EngramDeviceRegion hidden, EngramDeviceRegion query_rank, EngramDeviceRegion query_phases,
      EngramDeviceRegion latent, EngramDeviceRegion compressed_phases, const LayerCacheRegions& cache,
      EngramDeviceRegion key_prefix, EngramDeviceRegion candidates, EngramDeviceRegion error,
      std::uintptr_t stream) const;
 private:
  IndexerPlan() = default;
  std::array<CacheSegment, 11> segments_{};
  Sha256Digest config_sha256_;
  std::uint64_t bytes_ = 0;
  std::uint32_t tokens_ = 0, positions_ = 0, world_ = 0, rank_ = 0;
};
}  // namespace pih::deepseek_v41
