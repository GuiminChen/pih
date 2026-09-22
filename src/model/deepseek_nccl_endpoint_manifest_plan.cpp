#include "pih/model/deepseek_nccl_endpoint_manifest_plan.h"

namespace pih {

Result<DeepSeekNcclEndpointManifestPlan>
DeepSeekNcclEndpointManifestPlan::Create(
    std::uint64_t engine_epoch, std::uint64_t communicator_generation,
    std::uint64_t config_identity,
    std::span<const DeepSeekNcclRankEndpointIdentity> ranks,
    std::span<const DeepSeekNcclIssuedEdgeCapability> capabilities) {
  if (engine_epoch == 0 || communicator_generation == 0 ||
      config_identity == 0 || ranks.size() < 2 || ranks.size() > 4 ||
      capabilities.size() + 1 != ranks.size()) {
    return Status::InvalidArgument(
        "DeepSeek NCCL endpoint manifest topology is invalid");
  }
  for (std::uint32_t rank = 0; rank < ranks.size(); ++rank) {
    if (ranks[rank].rank != rank || ranks[rank].device_identity == 0 ||
        ranks[rank].context_identity == 0) {
      return Status::FailedPrecondition(
          "DeepSeek NCCL rank endpoint identity is invalid");
    }
  }
  std::vector<DeepSeekNcclEdgeManifestPair> edges;
  edges.reserve(capabilities.size());
  for (std::uint32_t edge = 0; edge < capabilities.size(); ++edge) {
    const auto& capability = capabilities[edge];
    if (capability.edge_id != edge || capability.lease_id == 0 ||
        capability.commitment_id == 0 || capability.lower == nullptr ||
        capability.upper == nullptr || capability.lower->zeroized() ||
        capability.upper->zeroized() ||
        capability.lower->commitment() != capability.upper->commitment()) {
      return Status::FailedPrecondition(
          "DeepSeek NCCL endpoint capability set is invalid");
    }
    const auto make = [&](std::uint32_t local, std::uint32_t peer,
                          std::uint32_t communicator_rank) {
      return DeepSeekNcclCommunicatorManifest{
          .engine_epoch = engine_epoch,
          .communicator_generation = communicator_generation,
          .bootstrap_lease_id = capability.lease_id,
          .bootstrap_commitment_id = capability.commitment_id,
          .edge_id = edge,
          .local_global_rank = local,
          .peer_global_rank = peer,
          .communicator_local_rank = communicator_rank,
          .device_identity = ranks[local].device_identity,
          .context_identity = ranks[local].context_identity,
          .config_identity = config_identity};
    };
    auto lower = make(edge, edge + 1, 0);
    auto upper = make(edge + 1, edge, 1);
    auto lower_valid = DeepSeekNcclCommunicator::Create(lower);
    auto upper_valid = DeepSeekNcclCommunicator::Create(upper);
    if (!lower_valid.ok()) return lower_valid.status();
    if (!upper_valid.ok()) return upper_valid.status();
    edges.push_back({lower, upper});
  }
  return DeepSeekNcclEndpointManifestPlan(
      static_cast<std::uint32_t>(ranks.size()), std::move(edges));
}

const DeepSeekNcclCommunicatorManifest*
DeepSeekNcclEndpointManifestPlan::incoming(std::uint32_t rank) const noexcept {
  return rank > 0 && rank < world_size_ ? &edges_[rank - 1].upper : nullptr;
}

const DeepSeekNcclCommunicatorManifest*
DeepSeekNcclEndpointManifestPlan::outgoing(std::uint32_t rank) const noexcept {
  return rank + 1 < world_size_ ? &edges_[rank].lower : nullptr;
}

}  // namespace pih
