#include "pih/model/catalog_placement_coordinator.h"

namespace pih {
namespace {

bool coordinator_inflight(CatalogPlacementState state) {
  // Only an exact route-withdrawal acknowledgement is currently a positive
  // release proof. ABORTED is reserved for a future durable reconciler.
  return state != CatalogPlacementState::kWithdrawn;
}

}  // namespace

CatalogPlacementCoordinator::CatalogPlacementCoordinator(
    std::size_t maximum_transactions) noexcept
    : maximum_transactions_(maximum_transactions) {
  transactions_.reserve(maximum_transactions);
}

Result<CatalogPlacementCoordinator> CatalogPlacementCoordinator::Create(
    std::size_t maximum_transactions) {
  if (maximum_transactions == 0 || maximum_transactions > 1024) {
    return Status::InvalidArgument(
        "catalog placement transaction bound is invalid");
  }
  return CatalogPlacementCoordinator(maximum_transactions);
}

Status CatalogPlacementCoordinator::begin(
    Sha256Digest,
    Sha256Digest,
    const RuntimeProfileSupervisorBootstrapManifest&) {
  return Status::FailedPrecondition(
      "catalog placement requires a frozen deadline policy");
}

CatalogPlacementTransaction* CatalogPlacementCoordinator::find(
    const Sha256Digest& transaction_id) {
  for (auto& transaction : transactions_) {
    if (transaction.transaction_id() == transaction_id) return &transaction;
  }
  return nullptr;
}

const CatalogPlacementTransaction* CatalogPlacementCoordinator::find(
    const Sha256Digest& transaction_id) const {
  for (const auto& transaction : transactions_) {
    if (transaction.transaction_id() == transaction_id) return &transaction;
  }
  return nullptr;
}

Status CatalogPlacementCoordinator::begin(
    Sha256Digest transaction_id,
    Sha256Digest target_topology_root,
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
    const CatalogPlacementDeadlinePolicy& deadlines,
    std::uint64_t begin_monotonic_ms) {
  if (find(transaction_id) != nullptr) {
    return Status::FailedPrecondition(
        "catalog placement transaction was replayed");
  }
  if (transactions_.size() == maximum_transactions_) {
    return Status::ResourceExhausted(
        "catalog placement transaction registry is full");
  }
  for (const auto& transaction : transactions_) {
    if (transaction.target_topology_root() == target_topology_root &&
        coordinator_inflight(transaction.state())) {
      return Status::FailedPrecondition(
          "catalog placement target already has an inflight transaction");
    }
  }
  auto transaction = CatalogPlacementTransaction::Create(
      transaction_id, target_topology_root, bootstrap, deadlines,
      begin_monotonic_ms);
  if (!transaction.ok()) return transaction.status();
  transactions_.push_back(std::move(*transaction));
  return Status::Ok();
}

Status CatalogPlacementCoordinator::mark_starting(
    const Sha256Digest& transaction_id) {
  auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition(
        "catalog placement transaction is unknown");
  }
  return transaction->mark_starting();
}

Status CatalogPlacementCoordinator::enforce_deadline(
    const Sha256Digest& transaction_id, std::uint64_t now_monotonic_ms) {
  auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition("catalog placement transaction is unknown");
  }
  return transaction->enforce_deadline(now_monotonic_ms);
}

Status CatalogPlacementCoordinator::quarantine(const Sha256Digest& transaction_id) {
  auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition("catalog placement transaction is unknown");
  }
  return transaction->quarantine();
}

Status CatalogPlacementCoordinator::bind_readiness(
    const Sha256Digest& transaction_id,
    const RuntimeProfileReadinessReceipt& receipt) {
  auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition(
        "catalog placement transaction is unknown");
  }
  return transaction->bind_readiness(receipt);
}

Status CatalogPlacementCoordinator::commit_route(
    const Sha256Digest& transaction_id,
    const Sha256Digest& active_authority_snapshot_root,
    const Sha256Digest& acknowledged_readiness_receipt_root) {
  auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition(
        "catalog placement transaction is unknown");
  }
  return transaction->commit_route(active_authority_snapshot_root,
                                   acknowledged_readiness_receipt_root);
}

Status CatalogPlacementCoordinator::begin_route_withdrawal(
    const Sha256Digest& transaction_id) {
  auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition(
        "catalog placement transaction is unknown");
  }
  return transaction->begin_route_withdrawal();
}

Status CatalogPlacementCoordinator::acknowledge_route_withdrawal(
    const Sha256Digest& transaction_id,
    const Sha256Digest& acknowledged_readiness_receipt_root) {
  auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition(
        "catalog placement transaction is unknown");
  }
  return transaction->acknowledge_route_withdrawal(
      acknowledged_readiness_receipt_root);
}

Status CatalogPlacementCoordinator::abort(
    const Sha256Digest& transaction_id) {
  auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition(
        "catalog placement transaction is unknown");
  }
  return transaction->abort();
}

Status CatalogPlacementCoordinator::fail(
    const Sha256Digest& transaction_id) {
  auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition(
        "catalog placement transaction is unknown");
  }
  return transaction->fail();
}

Result<CatalogPlacementState> CatalogPlacementCoordinator::state(
    const Sha256Digest& transaction_id) const {
  const auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition(
        "catalog placement transaction is unknown");
  }
  return transaction->state();
}

Result<CatalogPlacementRecoverySnapshot>
CatalogPlacementCoordinator::recovery_snapshot(
    const Sha256Digest& transaction_id) const {
  const auto* transaction = find(transaction_id);
  if (transaction == nullptr) {
    return Status::FailedPrecondition("catalog placement transaction is unknown");
  }
  return transaction->recovery_snapshot();
}

}  // namespace pih
