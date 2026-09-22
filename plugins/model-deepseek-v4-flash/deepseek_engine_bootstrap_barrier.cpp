#include "pih/model/deepseek_engine_bootstrap_barrier.h"

namespace pih {

Result<DeepSeekRankBootstrapReceipt> DeepSeekRankBootstrapReceipt::Create(
    std::uint64_t epoch, std::uint32_t world_size,
    const DeepSeekStageMappedInventory& inventory,
    const DeepSeekWeightMaterializationPlan& materialization,
    const DeepSeekWeightFinalizer& finalizer,
    DeepSeekRoutedExpertResidency expert_residency,
    const DeepSeekExpertPager* pager) {
  if (epoch == 0 || world_size == 0 || world_size > 4 ||
      inventory.rank() >= world_size || inventory.mapped_interval_bytes() == 0 ||
      materialization.owner_rank() != inventory.rank() ||
      materialization.backing_bytes() == 0 ||
      finalizer.state() != DeepSeekWeightFinalizeState::kSealed) {
    return Status::FailedPrecondition(
        "DeepSeek rank bootstrap receipt prerequisites are incomplete");
  }
  auto unchanged = finalizer.verify_unchanged();
  if (!unchanged.ok()) return unchanged;
  const bool host_spill =
      expert_residency == DeepSeekRoutedExpertResidency::kHostSpill;
  if ((host_spill &&
       (materialization.paged_source_bytes() == 0 || pager == nullptr ||
        pager->transfer_reservation_window() == 0 ||
        pager->slot_count() < pager->transfer_reservation_window() ||
        pager->staging_extent_count() <
            pager->transfer_reservation_window())) ||
      (!host_spill &&
       (materialization.paged_source_bytes() != 0 || pager != nullptr))) {
    return Status::FailedPrecondition(
        "DeepSeek rank expert residency resources are inconsistent");
  }
  DeepSeekRankBootstrapReceipt receipt;
  receipt.epoch_ = epoch;
  receipt.rank_ = inventory.rank();
  receipt.world_size_ = world_size;
  receipt.mapped_bytes_ = inventory.mapped_interval_bytes();
  receipt.fixed_backing_bytes_ = materialization.backing_bytes();
  receipt.paged_source_bytes_ = materialization.paged_source_bytes();
  receipt.expert_slot_count_ = pager == nullptr ? 0 : pager->slot_count();
  receipt.staging_extent_count_ =
      pager == nullptr ? 0 : pager->staging_extent_count();
  receipt.weight_layout_digest_ = finalizer.layout_digest();
  receipt.weight_seal_digest_ = finalizer.seal_digest();
  return receipt;
}

Result<DeepSeekEngineBootstrapBarrier> DeepSeekEngineBootstrapBarrier::Create(
    DeepSeekArtifactEpochGuard& artifact_guard) {
  if (artifact_guard.epoch() == 0 || artifact_guard.world_size() == 0 ||
      artifact_guard.world_size() > 4 ||
      artifact_guard.state() != ArtifactLeaseEpochState::kBootstrapping) {
    return Status::FailedPrecondition(
        "DeepSeek engine bootstrap artifact guard is invalid");
  }
  return DeepSeekEngineBootstrapBarrier(artifact_guard);
}

Status DeepSeekEngineBootstrapBarrier::accept(
    const DeepSeekRankBootstrapReceipt& receipt) {
  if (artifact_guard_->state() != ArtifactLeaseEpochState::kBootstrapping ||
      receipt.epoch() != artifact_guard_->epoch() ||
      receipt.world_size() != artifact_guard_->world_size() ||
      receipt.rank() >= accepted_.size() || accepted_[receipt.rank()]) {
    return Status::FailedPrecondition(
        "DeepSeek rank bootstrap receipt is stale or duplicated");
  }
  accepted_[receipt.rank()] = true;
  ++accepted_rank_count_;
  if (accepted_rank_count_ == accepted_.size()) {
    return artifact_guard_->mark_ready();
  }
  return Status::Ok();
}

bool DeepSeekEngineBootstrapBarrier::ready() const noexcept {
  return accepted_rank_count_ == accepted_.size() &&
         artifact_guard_->admission_allowed();
}

}  // namespace pih
