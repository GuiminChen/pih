#pragma once
#include "uploaded_bindings.h"
#include "sequence_cache.h"

namespace pih::deepseek_v41 {
class EngramPlan final {
 public:
  static Result<EngramPlan> Create(const FlashConfig& config, std::uint32_t token_capacity,
      std::uint32_t world, std::uint32_t rank, std::uint64_t budget);
  std::uint64_t bytes() const noexcept { return bytes_; }
  Result<EngramLaunch> Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
      EngramDeviceRegion arena, std::uint32_t layer, std::uint32_t tokens,
      EngramDeviceRegion hashes, EngramDeviceRegion residual, EngramDeviceRegion error, std::uintptr_t stream) const;
 private:
  EngramPlan() = default;
  std::array<CacheSegment, 5> segments_{};
  Sha256Digest config_sha256_;
  std::uint64_t bytes_ = 0;
  std::uint32_t tokens_ = 0, world_ = 0, rank_ = 0;
};
}  // namespace pih::deepseek_v41
