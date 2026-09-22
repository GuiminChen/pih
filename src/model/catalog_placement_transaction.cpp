#include "pih/model/catalog_placement_transaction.h"

namespace pih {
namespace {

bool placement_nonzero(const Sha256Digest& value) {
  std::byte aggregate{};
  for (const auto byte : value.bytes) aggregate |= byte;
  return aggregate != std::byte{};
}

bool placement_terminal(CatalogPlacementState state) {
  return state == CatalogPlacementState::kWithdrawn ||
         state == CatalogPlacementState::kAborted ||
         state == CatalogPlacementState::kFailed ||
         state == CatalogPlacementState::kQuarantined ||
         state == CatalogPlacementState::kAborting;
}

std::size_t deadline_index(CatalogPlacementState state) {
  switch (state) {
    case CatalogPlacementState::kVerifying: return 0;
    case CatalogPlacementState::kStarting: return 1;
    case CatalogPlacementState::kRouteCommitting: return 2;
    case CatalogPlacementState::kRouted:
    case CatalogPlacementState::kRouteWithdrawing: return 3;
    default: return 4;
  }
}

}  // namespace

CatalogPlacementTransaction::CatalogPlacementTransaction(
    Sha256Digest transaction_id,
    Sha256Digest target_topology_root,
    std::uint64_t deployment_generation,
    Sha256Digest authority_snapshot_root,
    Sha256Digest bootstrap_manifest_root,
    Sha256Digest readiness_nonce_digest,
    std::array<std::uint64_t, 5> absolute_deadlines,
    std::uint64_t begin_monotonic_ms) noexcept
    : transaction_id_(transaction_id),
      target_topology_root_(target_topology_root),
      deployment_generation_(deployment_generation),
      authority_snapshot_root_(authority_snapshot_root),
      bootstrap_manifest_root_(bootstrap_manifest_root),
      readiness_nonce_digest_(readiness_nonce_digest),
      absolute_deadlines_(absolute_deadlines),
      last_monotonic_ms_(begin_monotonic_ms) {}

Result<CatalogPlacementTransaction> CatalogPlacementTransaction::Create(
    Sha256Digest transaction_id,
    Sha256Digest target_topology_root,
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
    const CatalogPlacementDeadlinePolicy& deadlines,
    std::uint64_t begin_monotonic_ms) {
  if (!placement_nonzero(transaction_id) ||
      !placement_nonzero(target_topology_root) ||
      bootstrap.deployment_generation() == 0) {
    return Status::InvalidArgument("catalog placement identity is invalid");
  }
  auto absolute_deadlines = deadlines.absolute_deadlines(begin_monotonic_ms);
  if (!absolute_deadlines.ok()) return absolute_deadlines.status();
  return CatalogPlacementTransaction(
      transaction_id, target_topology_root, bootstrap.deployment_generation(),
      bootstrap.authority_snapshot_root(), bootstrap.manifest_root(),
      bootstrap.readiness_nonce_digest(), *absolute_deadlines,
      begin_monotonic_ms);
}

CatalogPlacementRecoverySnapshot
CatalogPlacementTransaction::recovery_snapshot() const noexcept {
  return CatalogPlacementRecoverySnapshot{
      transaction_id_, target_topology_root_, authority_snapshot_root_,
      bootstrap_manifest_root_, readiness_nonce_digest_, readiness_receipt_root_,
      absolute_deadlines_, deployment_generation_, state_};
}

Status CatalogPlacementTransaction::mark_starting() {
  if (state_ != CatalogPlacementState::kVerifying) {
    return Status::FailedPrecondition(
        "catalog placement cannot enter starting from current state");
  }
  state_ = CatalogPlacementState::kStarting;
  return Status::Ok();
}

Status CatalogPlacementTransaction::enforce_deadline(
    std::uint64_t now_monotonic_ms) {
  if (now_monotonic_ms == 0 || placement_terminal(state_) ||
      state_ == CatalogPlacementState::kAborting) {
    return Status::FailedPrecondition("catalog placement deadline state is invalid");
  }
  if (now_monotonic_ms < last_monotonic_ms_) {
    state_ = CatalogPlacementState::kAborting;
    return Status::FailedPrecondition("catalog placement monotonic clock regressed");
  }
  last_monotonic_ms_ = now_monotonic_ms;
  if (now_monotonic_ms >= absolute_deadlines_[deadline_index(state_)]) {
    state_ = CatalogPlacementState::kAborting;
    return Status::FailedPrecondition("catalog placement deadline expired");
  }
  return Status::Ok();
}

Status CatalogPlacementTransaction::quarantine() {
  if (state_ == CatalogPlacementState::kWithdrawn ||
      state_ == CatalogPlacementState::kAborted ||
      state_ == CatalogPlacementState::kFailed ||
      state_ == CatalogPlacementState::kQuarantined) {
    return Status::FailedPrecondition("catalog placement terminal state cannot be changed");
  }
  state_ = CatalogPlacementState::kQuarantined;
  return Status::Ok();
}

Status CatalogPlacementTransaction::bind_readiness(
    const RuntimeProfileReadinessReceipt& receipt) {
  if (state_ != CatalogPlacementState::kStarting ||
      receipt.deployment_generation() != deployment_generation_ ||
      !(receipt.authority_snapshot_root() == authority_snapshot_root_) ||
      !(receipt.bootstrap_manifest_root() == bootstrap_manifest_root_) ||
      !(receipt.readiness_nonce_digest() == readiness_nonce_digest_)) {
    return Status::FailedPrecondition(
        "catalog placement readiness does not match transaction");
  }
  readiness_receipt_root_ = receipt.receipt_root();
  state_ = CatalogPlacementState::kRouteCommitting;
  return Status::Ok();
}

Status CatalogPlacementTransaction::commit_route(
    const Sha256Digest& active_authority_snapshot_root,
    const Sha256Digest& acknowledged_readiness_receipt_root) {
  if (state_ != CatalogPlacementState::kRouteCommitting ||
      !(active_authority_snapshot_root == authority_snapshot_root_) ||
      !(acknowledged_readiness_receipt_root == readiness_receipt_root_)) {
    return Status::FailedPrecondition(
        "catalog placement route compare-and-match failed");
  }
  state_ = CatalogPlacementState::kRouted;
  return Status::Ok();
}

Status CatalogPlacementTransaction::begin_route_withdrawal() {
  if (state_ != CatalogPlacementState::kRouted) {
    return Status::FailedPrecondition(
        "catalog placement cannot begin route withdrawal from current state");
  }
  state_ = CatalogPlacementState::kRouteWithdrawing;
  return Status::Ok();
}

Status CatalogPlacementTransaction::acknowledge_route_withdrawal(
    const Sha256Digest& acknowledged_readiness_receipt_root) {
  if (state_ != CatalogPlacementState::kRouteWithdrawing ||
      !(acknowledged_readiness_receipt_root == readiness_receipt_root_)) {
    return Status::FailedPrecondition(
        "catalog placement route withdrawal compare-and-match failed");
  }
  state_ = CatalogPlacementState::kWithdrawn;
  return Status::Ok();
}

Status CatalogPlacementTransaction::abort() {
  if (placement_terminal(state_) || state_ == CatalogPlacementState::kRouted ||
      state_ == CatalogPlacementState::kRouteWithdrawing) {
    return Status::FailedPrecondition(
        "catalog placement terminal state cannot be changed");
  }
  // Local state cannot prove that routes, engine descendants, GPU leases, or
  // inherited descriptors are gone.  Do not release the target optimistically.
  state_ = CatalogPlacementState::kQuarantined;
  return Status::Ok();
}

Status CatalogPlacementTransaction::fail() {
  if (placement_terminal(state_) || state_ == CatalogPlacementState::kRouted ||
      state_ == CatalogPlacementState::kRouteWithdrawing) {
    return Status::FailedPrecondition(
        "catalog placement terminal state cannot be changed");
  }
  // See abort(): failures require durable external reconciliation before reuse.
  state_ = CatalogPlacementState::kQuarantined;
  return Status::Ok();
}

}  // namespace pih
