#pragma once

#include <memory>

#include "pih/model/deepseek_nccl_edge_endpoint.h"

namespace pih {

class DeepSeekRankBoundaryEndpoints final {
 public:
  static Status ValidateTopology(
      std::uint32_t rank, std::uint32_t world_size,
      const DeepSeekNcclCommunicatorManifest* incoming,
      const DeepSeekNcclCommunicatorManifest* outgoing);

  static Result<DeepSeekRankBoundaryEndpoints> Create(
      std::uint32_t rank, std::uint32_t world_size,
      std::unique_ptr<DeepSeekNcclEdgeEndpoint> incoming,
      std::unique_ptr<DeepSeekNcclEdgeEndpoint> outgoing);

  DeepSeekRankBoundaryEndpoints(const DeepSeekRankBoundaryEndpoints&) = delete;
  DeepSeekRankBoundaryEndpoints& operator=(
      const DeepSeekRankBoundaryEndpoints&) = delete;
  DeepSeekRankBoundaryEndpoints(DeepSeekRankBoundaryEndpoints&&) noexcept =
      default;
  DeepSeekRankBoundaryEndpoints& operator=(
      DeepSeekRankBoundaryEndpoints&&) noexcept = default;

  [[nodiscard]] DeepSeekBoundaryTransportDriver* incoming_transport()
      noexcept;
  [[nodiscard]] DeepSeekBoundaryTransportDriver* outgoing_transport()
      noexcept;
  [[nodiscard]] std::uint64_t communicator_generation() const noexcept {
    return communicator_generation_;
  }

 private:
  DeepSeekRankBoundaryEndpoints(
      std::unique_ptr<DeepSeekNcclEdgeEndpoint> incoming,
      std::unique_ptr<DeepSeekNcclEdgeEndpoint> outgoing,
      std::uint64_t communicator_generation) noexcept
      : incoming_(std::move(incoming)), outgoing_(std::move(outgoing)),
        communicator_generation_(communicator_generation) {}

  std::unique_ptr<DeepSeekNcclEdgeEndpoint> incoming_;
  std::unique_ptr<DeepSeekNcclEdgeEndpoint> outgoing_;
  std::uint64_t communicator_generation_ = 0;
};

}  // namespace pih
