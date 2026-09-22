#pragma once

#include <cstddef>
#include <vector>

#include "pih/model/catalog_placement_transaction.h"

namespace pih {

class CatalogPlacementCoordinator final {
 public:
  static Result<CatalogPlacementCoordinator> Create(
      std::size_t maximum_transactions);

  // Compatibility entry point: intentionally rejects unbound placement.
  Status begin(Sha256Digest transaction_id,
               Sha256Digest target_topology_root,
               const RuntimeProfileSupervisorBootstrapManifest& bootstrap);
  Status begin(Sha256Digest transaction_id,
               Sha256Digest target_topology_root,
               const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
               const CatalogPlacementDeadlinePolicy& deadlines,
               std::uint64_t begin_monotonic_ms);
  Status mark_starting(const Sha256Digest& transaction_id);
  Status enforce_deadline(const Sha256Digest& transaction_id,
                          std::uint64_t now_monotonic_ms);
  Status quarantine(const Sha256Digest& transaction_id);
  Status bind_readiness(const Sha256Digest& transaction_id,
                        const RuntimeProfileReadinessReceipt& receipt);
  Status commit_route(
      const Sha256Digest& transaction_id,
      const Sha256Digest& active_authority_snapshot_root,
      const Sha256Digest& acknowledged_readiness_receipt_root);
  Status begin_route_withdrawal(const Sha256Digest& transaction_id);
  Status acknowledge_route_withdrawal(
      const Sha256Digest& transaction_id,
      const Sha256Digest& acknowledged_readiness_receipt_root);
  Status abort(const Sha256Digest& transaction_id);
  Status fail(const Sha256Digest& transaction_id);
  Result<CatalogPlacementState> state(
      const Sha256Digest& transaction_id) const;
  Result<CatalogPlacementRecoverySnapshot> recovery_snapshot(
      const Sha256Digest& transaction_id) const;

 private:
  explicit CatalogPlacementCoordinator(
      std::size_t maximum_transactions) noexcept;
  CatalogPlacementTransaction* find(const Sha256Digest& transaction_id);
  const CatalogPlacementTransaction* find(
      const Sha256Digest& transaction_id) const;

  std::size_t maximum_transactions_ = 0;
  std::vector<CatalogPlacementTransaction> transactions_;
};

}  // namespace pih
