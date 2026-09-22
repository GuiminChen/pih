#pragma once
#include "token_embedding.h"
#include "token_input_upload.h"
#include "model_head.h"
#include "sampling.h"
#include "sequence_cache.h"

namespace pih::deepseek_v41 {
struct BoundaryViews final {
  TokenEmbeddingLaunch embedding;
  TokenInputUploadLaunch input_upload;
  ModelHeadLaunch head;
  SamplingLaunch sampling;
  EngramDeviceRegion logits, host_error, host_counts, host_candidate;
};
// Allocates distinct segments, not overlapping-lifetime reuse. Addresses stay
// stable when a smaller token prefix (e.g. decode) is bound to the same plan.
class BoundaryPlan final {
 public:
  static Result<BoundaryPlan> Create(const FlashConfig& config, std::uint32_t token_capacity,
      std::uint32_t world, std::uint32_t rank, std::uint64_t device_budget, std::uint64_t host_budget);
  std::uint64_t device_bytes() const noexcept { return device_bytes_; }
  std::uint64_t host_bytes() const noexcept { return host_bytes_; }
  std::uint32_t token_capacity() const noexcept { return tokens_; }
  std::uint32_t world_size() const noexcept { return world_; }
  std::uint32_t rank() const noexcept { return rank_; }
  const Sha256Digest& config_sha256() const noexcept { return config_sha256_; }
  // Device and pinned-host arena identity/lifetime are the owner's obligation.
  // Weight placeholders are filled by the execution entries; head residual/pre
  // are replaced with the completed sequence output by HeadOperation.
  Result<BoundaryViews> Bind(EngramDeviceRegion device, EngramDeviceRegion host,
      std::uint32_t tokens, std::uintptr_t stream) const;
 private:
  BoundaryPlan() = default;
  std::array<CacheSegment, 17> device_{};
  std::array<CacheSegment, 6> host_{};
  std::uint64_t device_bytes_ = 0, host_bytes_ = 0;
  std::uint32_t tokens_ = 0, world_ = 0, rank_ = 0;
  Sha256Digest config_sha256_;
};
}  // namespace pih::deepseek_v41
