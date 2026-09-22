#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_nccl_bootstrap_capability_issuer.h"
#include "pih/model/deepseek_nccl_communicator.h"

namespace pih {

struct DeepSeekNcclRankEndpointIdentity final {
  std::uint32_t rank = 0;
  std::uint64_t device_identity = 0;
  std::uintptr_t context_identity = 0;
};

struct DeepSeekNcclEdgeManifestPair final {
  DeepSeekNcclCommunicatorManifest lower;
  DeepSeekNcclCommunicatorManifest upper;
};

class DeepSeekNcclEndpointManifestPlan final {
 public:
  static Result<DeepSeekNcclEndpointManifestPlan> Create(
      std::uint64_t engine_epoch, std::uint64_t communicator_generation,
      std::uint64_t config_identity,
      std::span<const DeepSeekNcclRankEndpointIdentity> ranks,
      std::span<const DeepSeekNcclIssuedEdgeCapability> capabilities);

  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] const std::vector<DeepSeekNcclEdgeManifestPair>& edges()
      const noexcept { return edges_; }
  [[nodiscard]] const DeepSeekNcclCommunicatorManifest* incoming(
      std::uint32_t rank) const noexcept;
  [[nodiscard]] const DeepSeekNcclCommunicatorManifest* outgoing(
      std::uint32_t rank) const noexcept;

 private:
  DeepSeekNcclEndpointManifestPlan(
      std::uint32_t world_size,
      std::vector<DeepSeekNcclEdgeManifestPair> edges) noexcept
      : world_size_(world_size), edges_(std::move(edges)) {}

  std::uint32_t world_size_ = 0;
  std::vector<DeepSeekNcclEdgeManifestPair> edges_;
};

}  // namespace pih
