#include "pih/model/deepseek_nccl_boundary_warmup_gate.h"

namespace pih {
namespace {

bool endpoint_matches(const DeepSeekNcclEndpointBootstrapReceipt& receipt,
                      const DeepSeekNcclCommunicatorManifest& expected) {
  return receipt.engine_epoch == expected.engine_epoch &&
         receipt.communicator_generation == expected.communicator_generation &&
         receipt.bootstrap_lease_id == expected.bootstrap_lease_id &&
         receipt.bootstrap_commitment_id == expected.bootstrap_commitment_id &&
         receipt.edge_id == expected.edge_id &&
         receipt.local_global_rank == expected.local_global_rank &&
         receipt.peer_global_rank == expected.peer_global_rank &&
         receipt.communicator_local_rank == expected.communicator_local_rank &&
         receipt.device_identity == expected.device_identity &&
         receipt.context_identity == expected.context_identity &&
         receipt.config_identity == expected.config_identity &&
         receipt.bootstrap_zeroized && receipt.communicator_reconciled;
}

}  // namespace

Result<DeepSeekNcclBoundaryWarmupGate>
DeepSeekNcclBoundaryWarmupGate::Create(
    const DeepSeekNcclEndpointManifestPlan& plan,
    std::uint32_t maximum_token_count) {
  if (plan.world_size() < 2 || plan.world_size() > 4 ||
      plan.edges().size() + 1 != plan.world_size() ||
      maximum_token_count == 0) {
    return Status::InvalidArgument(
        "DeepSeek NCCL boundary warm-up gate identity is invalid");
  }
  return DeepSeekNcclBoundaryWarmupGate(plan.edges(), maximum_token_count);
}

Status DeepSeekNcclBoundaryWarmupGate::poison(Status cause) noexcept {
  poisoned_ = true;
  return cause.ok() ? Status::Internal(
                          "DeepSeek NCCL boundary warm-up gate poisoned")
                    : cause;
}

Status DeepSeekNcclBoundaryWarmupGate::accept(
    DeepSeekNcclEndpointWarmupReceipt receipt) {
  if (poisoned_ || finalized_) {
    return Status::Unavailable(
        "DeepSeek NCCL boundary warm-up gate is closed");
  }
  const auto edge = receipt.endpoint.edge_id;
  const auto local = receipt.endpoint.communicator_local_rank;
  if (edge >= expected_.size() || local > 1) {
    return poison(Status::InvalidArgument(
        "DeepSeek NCCL warm-up receipt topology is invalid"));
  }
  const auto index = static_cast<std::size_t>(edge) * 2 + local;
  if (receipts_[index].has_value()) {
    return poison(Status::FailedPrecondition(
        "DeepSeek NCCL warm-up receipt was replayed"));
  }
  const auto& expected = local == 0 ? expected_[edge].lower
                                   : expected_[edge].upper;
  if (!endpoint_matches(receipt.endpoint, expected) ||
      receipt.minimum_token_count != 1 ||
      receipt.maximum_token_count != maximum_token_count_ ||
      !receipt.minimum_complete_verified ||
      !receipt.maximum_complete_verified) {
    return poison(Status::FailedPrecondition(
        "DeepSeek NCCL warm-up receipt evidence is invalid"));
  }
  receipts_[index] = std::move(receipt);
  ++accepted_count_;
  return Status::Ok();
}

Result<std::vector<DeepSeekNcclWarmupReceipt>>
DeepSeekNcclBoundaryWarmupGate::finalize() {
  if (poisoned_ || finalized_ || accepted_count_ != receipts_.size()) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL boundary warm-up evidence is incomplete");
  }
  std::vector<DeepSeekNcclWarmupReceipt> result;
  result.reserve(expected_.size());
  for (std::size_t edge = 0; edge < expected_.size(); ++edge) {
    const auto& lower = *receipts_[edge * 2];
    const auto& upper = *receipts_[edge * 2 + 1];
    if (lower.minimum_payload_digest != upper.minimum_payload_digest ||
        lower.maximum_payload_digest != upper.maximum_payload_digest) {
      return poison(Status::Internal(
          "DeepSeek NCCL warm-up payload differs across edge endpoints"));
    }
    result.push_back(
        {expected_[edge].lower.engine_epoch,
         expected_[edge].lower.communicator_generation,
         static_cast<std::uint32_t>(edge), true, true});
  }
  finalized_ = true;
  return result;
}

}  // namespace pih
