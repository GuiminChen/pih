#pragma once
#include "uploaded_bindings.h"
#include "attention_residual.h"
#include "sequence_cache.h"

namespace pih::deepseek_v41 {
class AttentionOutputPlan final {
 public:
  static Result<AttentionOutputPlan> Create(const FlashConfig& config, std::uint32_t token_capacity,
      std::uint32_t maximum_positions, std::uint32_t world, std::uint32_t rank, std::uint64_t budget);
  std::uint64_t bytes() const noexcept { return bytes_; }
  Result<AttentionResidualLaunch> Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
      EngramDeviceRegion arena, const AttentionPrepareLaunch& source,
      EngramDeviceRegion compressed_prefix, EngramDeviceRegion selected) const;
 private:
  AttentionOutputPlan() = default;
  std::array<CacheSegment, 10> segments_{};
  Sha256Digest config_sha256_;
  std::uint64_t bytes_ = 0;
  std::uint32_t tokens_ = 0, maximum_ = 0, world_ = 0, rank_ = 0;
};
}  // namespace pih::deepseek_v41
