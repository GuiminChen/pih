#pragma once
#include "uploaded_bindings.h"
#include "sequence_cache.h"

namespace pih::deepseek_v41 {
// Borrowed transient arena; persistent pooling state and KV storage belong to
// SequenceCachePlan. The input is the normalized attention hidden state.
class CompressorPlan final {
 public:
  static Result<CompressorPlan> Create(const FlashConfig& config,
      std::uint32_t token_capacity, std::uint64_t byte_budget);
  std::uint64_t bytes() const noexcept { return bytes_; }
  Result<CompressedPrepareLaunch> Bind(const BackboneWeightUpload& weights,
      const FlashConfig& config, EngramDeviceRegion arena, std::uint32_t layer,
      std::uint32_t start, std::uint32_t tokens, EngramDeviceRegion hidden,
      EngramDeviceRegion phases, const LayerCacheRegions& cache,
      EngramDeviceRegion error, std::uintptr_t stream) const;
 private:
  CompressorPlan() = default;
  std::array<CacheSegment, 5> segments_{};
  Sha256Digest config_sha256_;
  std::uint64_t bytes_ = 0;
  std::uint32_t tokens_ = 0;
};
}  // namespace pih::deepseek_v41
