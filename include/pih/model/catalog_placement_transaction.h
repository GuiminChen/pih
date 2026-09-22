#pragma once

#include "pih/model/catalog_placement_deadline_policy.h"
#include "pih/model/runtime_profile_readiness_receipt.h"

namespace pih {

enum class CatalogPlacementState : std::uint8_t {
  kVerifying = 1,
  kStarting = 2,
  kRouteCommitting = 3,
  kRouted = 4,
  kAborted = 5,
  kFailed = 6,
  kRouteWithdrawing = 7,
  kWithdrawn = 8,
  // A local controller has observed failure but lacks external cleanup proof.
  kQuarantined = 9,
  kAborting = 10,
};

struct CatalogPlacementRecoverySnapshot final {
  Sha256Digest transaction_id{};
  Sha256Digest target_topology_root{};
  Sha256Digest authority_snapshot_root{};
  Sha256Digest bootstrap_manifest_root{};
  Sha256Digest readiness_nonce_digest{};
  Sha256Digest readiness_receipt_root{};
  std::array<std::uint64_t, 5> absolute_deadlines{};
  std::uint64_t deployment_generation = 0;
  CatalogPlacementState state = CatalogPlacementState::kQuarantined;
};

class CatalogPlacementTransaction final {
 public:
  static Result<CatalogPlacementTransaction> Create(
      Sha256Digest transaction_id,
      Sha256Digest target_topology_root,
      const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
      const CatalogPlacementDeadlinePolicy& deadlines,
      std::uint64_t begin_monotonic_ms);

  [[nodiscard]] CatalogPlacementState state() const noexcept { return state_; }
  [[nodiscard]] const Sha256Digest& transaction_id() const noexcept {
    return transaction_id_;
  }
  [[nodiscard]] const Sha256Digest& target_topology_root() const noexcept {
    return target_topology_root_;
  }
  [[nodiscard]] const Sha256Digest& readiness_receipt_root() const noexcept {
    return readiness_receipt_root_;
  }
  [[nodiscard]] const std::array<std::uint64_t, 5>& absolute_deadlines() const noexcept {
    return absolute_deadlines_;
  }
  [[nodiscard]] CatalogPlacementRecoverySnapshot recovery_snapshot() const noexcept;

  Status mark_starting();
  Status enforce_deadline(std::uint64_t now_monotonic_ms);
  Status quarantine();
  Status bind_readiness(const RuntimeProfileReadinessReceipt& receipt);
  Status commit_route(const Sha256Digest& active_authority_snapshot_root,
                      const Sha256Digest& acknowledged_readiness_receipt_root);
  Status begin_route_withdrawal();
  Status acknowledge_route_withdrawal(
      const Sha256Digest& acknowledged_readiness_receipt_root);
  Status abort();
  Status fail();

 private:
  CatalogPlacementTransaction(
      Sha256Digest transaction_id,
      Sha256Digest target_topology_root,
      std::uint64_t deployment_generation,
      Sha256Digest authority_snapshot_root,
      Sha256Digest bootstrap_manifest_root,
      Sha256Digest readiness_nonce_digest,
      std::array<std::uint64_t, 5> absolute_deadlines,
      std::uint64_t begin_monotonic_ms) noexcept;

  Sha256Digest transaction_id_{};
  Sha256Digest target_topology_root_{};
  std::uint64_t deployment_generation_ = 0;
  Sha256Digest authority_snapshot_root_{};
  Sha256Digest bootstrap_manifest_root_{};
  Sha256Digest readiness_nonce_digest_{};
  Sha256Digest readiness_receipt_root_{};
  std::array<std::uint64_t, 5> absolute_deadlines_{};
  std::uint64_t last_monotonic_ms_ = 0;
  CatalogPlacementState state_ = CatalogPlacementState::kVerifying;
};

}  // namespace pih
