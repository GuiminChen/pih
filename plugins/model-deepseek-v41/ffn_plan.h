#pragma once
#include "uploaded_bindings.h"
#include "ffn_route.h"
#include "expert_residual.h"
#include "sequence_cache.h"

namespace pih::deepseek_v41 {
struct FfnViews final { FfnRouteLaunch route; ExpertResidualLaunch tail; };
class FfnPlan final {
 public:
  static Result<FfnPlan> Create(const FlashConfig& config, std::uint32_t token_capacity,
      std::uint32_t world, std::uint32_t rank, std::uint64_t byte_budget);
  std::uint64_t bytes() const noexcept { return bytes_; }
  std::uint32_t world_size() const noexcept { return world_; }
  std::uint32_t rank() const noexcept { return rank_; }
  // Constructs and weight-binds the complete FFN descriptors. The accumulator
  // belongs to the separately allocated routed-expert workspace; its contents
  // are produced/reduced later by FfnOperation, not admitted by this builder.
  Result<FfnViews> Bind(const BackboneWeightUpload& weights, const FlashConfig& config,
      EngramDeviceRegion arena, std::uint32_t layer, std::uint32_t tokens,
      EngramDeviceRegion residual, EngramDeviceRegion carried_pre,
      EngramDeviceRegion accumulator, EngramDeviceRegion error, std::uintptr_t stream) const;
 private:
  FfnPlan() = default;
  std::array<CacheSegment, 22> segments_{};
  Sha256Digest config_sha256_;
  std::uint64_t bytes_ = 0;
  std::uint32_t tokens_ = 0, world_ = 0, rank_ = 0;
};
}  // namespace pih::deepseek_v41
