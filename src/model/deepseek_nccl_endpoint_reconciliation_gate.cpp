#include "pih/model/deepseek_nccl_endpoint_reconciliation_gate.h"

namespace pih {
namespace {

bool matches(const DeepSeekNcclEndpointBootstrapReceipt& receipt,
             const DeepSeekNcclCommunicatorManifest& expected) noexcept {
  return receipt.engine_epoch == expected.engine_epoch &&
         receipt.communicator_generation ==
             expected.communicator_generation &&
         receipt.bootstrap_lease_id == expected.bootstrap_lease_id &&
         receipt.bootstrap_commitment_id ==
             expected.bootstrap_commitment_id &&
         receipt.edge_id == expected.edge_id &&
         receipt.local_global_rank == expected.local_global_rank &&
         receipt.peer_global_rank == expected.peer_global_rank &&
         receipt.communicator_local_rank ==
             expected.communicator_local_rank &&
         receipt.device_identity == expected.device_identity &&
         receipt.context_identity == expected.context_identity &&
         receipt.config_identity == expected.config_identity &&
         receipt.bootstrap_zeroized && receipt.communicator_reconciled;
}

}  // namespace

Result<DeepSeekNcclEndpointBootstrapReceipt>
DeepSeekNcclEndpointBootstrapReceipt::Create(
    const DeepSeekNcclEdgeEndpoint& endpoint) {
  if (endpoint.state() != DeepSeekNcclCommunicatorState::kReconciled ||
      !endpoint.bootstrap_zeroized()) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL endpoint is not reconciled for receipt publication");
  }
  const auto& manifest = endpoint.manifest();
  return DeepSeekNcclEndpointBootstrapReceipt{
      manifest.engine_epoch,
      manifest.communicator_generation,
      manifest.bootstrap_lease_id,
      manifest.bootstrap_commitment_id,
      manifest.edge_id,
      manifest.local_global_rank,
      manifest.peer_global_rank,
      manifest.communicator_local_rank,
      manifest.device_identity,
      manifest.context_identity,
      manifest.config_identity,
      true,
      true};
}

Result<DeepSeekNcclEndpointReconciliationGate>
DeepSeekNcclEndpointReconciliationGate::Create(
    const DeepSeekNcclEndpointManifestPlan& plan) {
  if (plan.world_size() < 2 || plan.world_size() > 4 ||
      plan.edges().size() + 1 != plan.world_size()) {
    return Status::InvalidArgument(
        "DeepSeek NCCL reconciliation gate plan is invalid");
  }
  return DeepSeekNcclEndpointReconciliationGate(plan.edges());
}

Status DeepSeekNcclEndpointReconciliationGate::poison(
    Status cause) noexcept {
  poisoned_ = true;
  return cause.ok() ? Status::Internal(
                          "DeepSeek NCCL reconciliation gate poisoned")
                    : cause;
}

Status DeepSeekNcclEndpointReconciliationGate::accept(
    const DeepSeekNcclEndpointBootstrapReceipt& receipt) {
  if (poisoned_) {
    return Status::Unavailable(
        "DeepSeek NCCL reconciliation gate is poisoned");
  }
  if (receipt.edge_id >= expected_.size() ||
      receipt.communicator_local_rank > 1) {
    return poison(Status::InvalidArgument(
        "DeepSeek NCCL endpoint receipt topology is invalid"));
  }
  const auto index = static_cast<std::size_t>(receipt.edge_id) * 2 +
                     receipt.communicator_local_rank;
  if (accepted_[index]) {
    return poison(Status::FailedPrecondition(
        "DeepSeek NCCL endpoint receipt was replayed"));
  }
  const auto& pair = expected_[receipt.edge_id];
  const auto& expected = receipt.communicator_local_rank == 0
                             ? pair.lower
                             : pair.upper;
  if (!matches(receipt, expected)) {
    return poison(Status::FailedPrecondition(
        "DeepSeek NCCL endpoint receipt differs from controller plan"));
  }
  accepted_[index] = true;
  ++accepted_count_;
  return Status::Ok();
}

}  // namespace pih
