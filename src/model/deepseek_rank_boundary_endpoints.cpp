#include "pih/model/deepseek_rank_boundary_endpoints.h"

#include <utility>

namespace pih {
namespace {

Status validate_endpoint(const DeepSeekNcclCommunicatorManifest& manifest,
                         std::uint32_t local, std::uint32_t peer,
                         std::uint32_t edge,
                         std::uint32_t communicator_rank) {
  if (manifest.local_global_rank != local ||
      manifest.peer_global_rank != peer || manifest.edge_id != edge ||
      manifest.communicator_local_rank != communicator_rank) {
    return Status::FailedPrecondition(
        "DeepSeek rank boundary endpoint topology is inconsistent");
  }
  return Status::Ok();
}

}  // namespace

Status DeepSeekRankBoundaryEndpoints::ValidateTopology(
    std::uint32_t rank, std::uint32_t world_size,
    const DeepSeekNcclCommunicatorManifest* incoming,
    const DeepSeekNcclCommunicatorManifest* outgoing) {
  if (world_size == 0 || world_size > 4 || rank >= world_size) {
    return Status::InvalidArgument(
        "DeepSeek rank boundary topology identity is invalid");
  }
  const bool needs_incoming = rank > 0;
  const bool needs_outgoing = rank + 1 < world_size;
  if ((incoming != nullptr) != needs_incoming ||
      (outgoing != nullptr) != needs_outgoing) {
    return Status::FailedPrecondition(
        "DeepSeek rank boundary endpoint count does not match topology");
  }
  if (incoming != nullptr) {
    auto status = validate_endpoint(*incoming, rank, rank - 1, rank - 1, 1);
    if (!status.ok()) return status;
  }
  if (outgoing != nullptr) {
    auto status = validate_endpoint(*outgoing, rank, rank + 1, rank, 0);
    if (!status.ok()) return status;
  }
  if (incoming != nullptr && outgoing != nullptr &&
      (incoming->engine_epoch != outgoing->engine_epoch ||
       incoming->communicator_generation !=
           outgoing->communicator_generation ||
       incoming->config_identity != outgoing->config_identity)) {
    return Status::FailedPrecondition(
        "DeepSeek rank boundary endpoints mix generation identity");
  }
  return Status::Ok();
}

Result<DeepSeekRankBoundaryEndpoints> DeepSeekRankBoundaryEndpoints::Create(
    std::uint32_t rank, std::uint32_t world_size,
    std::unique_ptr<DeepSeekNcclEdgeEndpoint> incoming,
    std::unique_ptr<DeepSeekNcclEdgeEndpoint> outgoing) {
  const auto* incoming_manifest = incoming == nullptr ? nullptr
                                                       : &incoming->manifest();
  const auto* outgoing_manifest = outgoing == nullptr ? nullptr
                                                       : &outgoing->manifest();
  const auto topology = ValidateTopology(rank, world_size, incoming_manifest,
                                         outgoing_manifest);
  if (!topology.ok()) return topology;
  if ((incoming != nullptr && incoming->transport() == nullptr) ||
      (outgoing != nullptr && outgoing->transport() == nullptr)) {
    return Status::FailedPrecondition(
        "DeepSeek rank boundary endpoints must be sealed before ownership");
  }
  const std::uint64_t generation =
      incoming_manifest != nullptr
          ? incoming_manifest->communicator_generation
          : (outgoing_manifest != nullptr
                 ? outgoing_manifest->communicator_generation
                 : 0);
  return DeepSeekRankBoundaryEndpoints(std::move(incoming),
                                       std::move(outgoing), generation);
}

DeepSeekBoundaryTransportDriver*
DeepSeekRankBoundaryEndpoints::incoming_transport() noexcept {
  return incoming_ == nullptr ? nullptr : incoming_->transport();
}

DeepSeekBoundaryTransportDriver*
DeepSeekRankBoundaryEndpoints::outgoing_transport() noexcept {
  return outgoing_ == nullptr ? nullptr : outgoing_->transport();
}

}  // namespace pih
