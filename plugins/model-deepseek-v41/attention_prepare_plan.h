#pragma once
#include "uploaded_bindings.h"
#include "sequence_cache.h"

namespace pih::deepseek_v41 {
class AttentionPreparePlan final {
 public:
  static Result<AttentionPreparePlan> Create(const FlashConfig& config, std::uint32_t token_capacity,
      std::uint32_t world, std::uint32_t rank, std::uint64_t byte_budget);
  std::uint64_t bytes() const noexcept { return bytes_; }
  Result<AttentionPrepareLaunch> Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
      EngramDeviceRegion arena, std::uint32_t layer, std::uint32_t start, std::uint32_t tokens,
      EngramDeviceRegion residual, EngramDeviceRegion pre, EngramDeviceRegion phases,
      EngramDeviceRegion window_ring, EngramDeviceRegion error, std::uintptr_t stream) const;
 private:
  AttentionPreparePlan() = default;
  std::array<CacheSegment, 18> segments_{};
  Sha256Digest config_sha256_;
  std::uint64_t bytes_ = 0;
  std::uint32_t tokens_ = 0, world_ = 0, rank_ = 0;
};
}  // namespace pih::deepseek_v41
