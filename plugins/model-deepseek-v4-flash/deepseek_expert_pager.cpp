#include "pih/model/deepseek_expert_pager.h"

#include "pih/model/deepseek_pager_cost_ledger.h"

namespace pih {

Result<DeepSeekExpertPager> DeepSeekExpertPager::Create(
    DeepSeekStageRange owned_layers, std::uint32_t slots,
    std::uint32_t staging_extents,
    std::uint32_t transfer_reservation_window) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer >= 43) {
    return Status::InvalidArgument("DeepSeek pager layer ownership is invalid");
  }
  if (transfer_reservation_window == 0 ||
      slots < transfer_reservation_window ||
      staging_extents < transfer_reservation_window) {
    return Status::InvalidArgument(
        "DeepSeek paging reservation window exceeds resource capacity");
  }
  DeepSeekExpertPager pager;
  pager.owned_layers_ = owned_layers;
  pager.slots_.resize(slots);
  pager.staging_extents_ = staging_extents;
  pager.transfer_reservation_window_ = transfer_reservation_window;
  return pager;
}

Result<DeepSeekExpertDemand> DeepSeekExpertPager::demand(
    DeepSeekExpertIdentity identity) {
  if (poisoned_) {
    return Status::Unavailable("DeepSeek expert pager epoch is poisoned");
  }
  if (identity.layer < owned_layers_.first_layer ||
      identity.layer > owned_layers_.last_layer || identity.expert >= 256) {
    return Status::InvalidArgument("DeepSeek expert is not owned by this stage");
  }
  for (std::uint32_t i = 0; i < slots_.size(); ++i) {
    auto& slot = slots_[i];
    if (slot.identity == identity) {
      if (slot.state == DeepSeekExpertSlotState::kResident) {
        return DeepSeekExpertDemand{
            DeepSeekExpertDemandDisposition::kResidentHit, i, slot.generation};
      }
      if (slot.state == DeepSeekExpertSlotState::kReserved ||
          slot.state == DeepSeekExpertSlotState::kH2dInFlight) {
        return DeepSeekExpertDemand{
            DeepSeekExpertDemandDisposition::kInflightJoin, i, slot.generation};
      }
    }
  }
  if (transfer_reservations_ >= transfer_reservation_window_ ||
      transfer_reservations_ >= staging_extents_) {
    return Status::ResourceExhausted(
        "DeepSeek transfer reservation window is full");
  }
  for (std::uint32_t i = 0; i < slots_.size(); ++i) {
    auto& slot = slots_[i];
    if (slot.state != DeepSeekExpertSlotState::kAbsent) continue;
    if (next_generation_ == 0) {
      return Status::ResourceExhausted("DeepSeek expert generation exhausted");
    }
    slot.identity = identity;
    slot.generation = next_generation_++;
    slot.state = DeepSeekExpertSlotState::kReserved;
    ++transfer_reservations_;
    return DeepSeekExpertDemand{DeepSeekExpertDemandDisposition::kNewMiss, i,
                                slot.generation};
  }
  return Status::ResourceExhausted("DeepSeek expert slots are not evictable");
}

Status DeepSeekExpertPager::begin_h2d(DeepSeekExpertIdentity identity,
                                      std::uint64_t generation,
                                      std::uint64_t payload_bytes) {
  if (poisoned_) {
    return Status::Unavailable("DeepSeek expert pager epoch is poisoned");
  }
  if (payload_bytes != kBundleBytes) {
    return Status::InvalidArgument("DeepSeek expert payload is not canonical");
  }
  for (auto& slot : slots_) {
    if (slot.identity == identity && slot.generation == generation &&
        slot.state == DeepSeekExpertSlotState::kReserved) {
      slot.state = DeepSeekExpertSlotState::kH2dInFlight;
      return Status::Ok();
    }
  }
  return Status::FailedPrecondition("DeepSeek expert reservation is stale");
}

Status DeepSeekExpertPager::complete_h2d(DeepSeekExpertIdentity identity,
                                         std::uint64_t generation) {
  if (poisoned_) {
    return Status::Unavailable("DeepSeek expert pager epoch is poisoned");
  }
  for (auto& slot : slots_) {
    if (slot.identity == identity && slot.generation == generation &&
        slot.state == DeepSeekExpertSlotState::kH2dInFlight) {
      if (transfer_reservations_ == 0) {
        slot.state = DeepSeekExpertSlotState::kFailed;
        poisoned_ = true;
        return Status::Internal(
            "DeepSeek expert transfer reservation accounting underflow");
      }
      slot.state = DeepSeekExpertSlotState::kResident;
      --transfer_reservations_;
      return Status::Ok();
    }
  }
  return Status::FailedPrecondition("DeepSeek expert transfer is stale");
}

Result<DeepSeekExpertLease> DeepSeekExpertPager::acquire(
    DeepSeekExpertIdentity identity, std::uint64_t generation) {
  if (poisoned_) {
    return Status::Unavailable("DeepSeek expert pager epoch is poisoned");
  }
  for (std::uint32_t i = 0; i < slots_.size(); ++i) {
    auto& slot = slots_[i];
    if (slot.identity == identity && slot.generation == generation &&
        slot.state == DeepSeekExpertSlotState::kResident) {
      if (slot.leases == UINT32_MAX) {
        poisoned_ = true;
        return Status::ResourceExhausted(
            "DeepSeek expert lease generation is exhausted");
      }
      ++slot.leases;
      return DeepSeekExpertLease{identity, i, generation};
    }
  }
  return Status::FailedPrecondition("DeepSeek expert is not leaseable");
}

Status DeepSeekExpertPager::release(const DeepSeekExpertLease& lease) {
  if (lease.slot >= slots_.size()) {
    return Status::InvalidArgument("DeepSeek expert lease slot is invalid");
  }
  auto& slot = slots_[lease.slot];
  if (slot.identity != lease.identity || slot.generation != lease.generation ||
      slot.leases == 0) {
    return Status::FailedPrecondition("DeepSeek expert lease is stale");
  }
  --slot.leases;
  if (slot.leases == 0 && slot.state == DeepSeekExpertSlotState::kEvictPending) {
    slot.state = DeepSeekExpertSlotState::kAbsent;
    slot.identity.reset();
  }
  return Status::Ok();
}

Status DeepSeekExpertPager::request_eviction(DeepSeekExpertIdentity identity,
                                             std::uint64_t generation) {
  if (poisoned_) {
    return Status::Unavailable("DeepSeek expert pager epoch is poisoned");
  }
  for (auto& slot : slots_) {
    if (slot.identity == identity && slot.generation == generation &&
        slot.state == DeepSeekExpertSlotState::kResident) {
      if (slot.leases == 0) {
        slot.state = DeepSeekExpertSlotState::kAbsent;
        slot.identity.reset();
      } else {
        slot.state = DeepSeekExpertSlotState::kEvictPending;
      }
      return Status::Ok();
    }
  }
  return Status::FailedPrecondition("DeepSeek expert eviction is stale");
}

Status DeepSeekExpertPager::fail_transfer(DeepSeekExpertIdentity identity,
                                          std::uint64_t generation) {
  if (poisoned_) return Status::Ok();
  for (auto& slot : slots_) {
    if (slot.identity == identity && slot.generation == generation &&
        (slot.state == DeepSeekExpertSlotState::kReserved ||
         slot.state == DeepSeekExpertSlotState::kH2dInFlight)) {
      slot.state = DeepSeekExpertSlotState::kFailed;
      poisoned_ = true;
      if (transfer_reservations_ == 0) {
        return Status::Internal(
            "DeepSeek failed transfer reservation accounting underflow");
      }
      --transfer_reservations_;
      return Status::Ok();
    }
  }
  return Status::FailedPrecondition("DeepSeek failed transfer is stale");
}

Result<std::uint64_t> DeepSeekExpertPager::debit_observed_cost(
    DeepSeekPagerCostLedger& ledger, DeepSeekPagerCostKey key,
    const DeepSeekExpertDemand& demand, std::uint64_t observed_cost_ns) {
  if (demand.slot >= slots_.size() || key.layer >= 43 || key.expert >= 256) {
    return Status::InvalidArgument(
        "DeepSeek observed pager cost binding is invalid");
  }
  const auto& slot = slots_[demand.slot];
  const DeepSeekExpertIdentity identity{key.layer, key.expert};
  if (slot.identity != identity || slot.generation != demand.generation) {
    return Status::FailedPrecondition(
        "DeepSeek observed pager cost demand is stale");
  }
  const bool disposition_matches =
      (demand.disposition == DeepSeekExpertDemandDisposition::kNewMiss &&
       (slot.state == DeepSeekExpertSlotState::kReserved ||
        slot.state == DeepSeekExpertSlotState::kH2dInFlight ||
        slot.state == DeepSeekExpertSlotState::kResident)) ||
      (demand.disposition == DeepSeekExpertDemandDisposition::kInflightJoin &&
       (slot.state == DeepSeekExpertSlotState::kReserved ||
        slot.state == DeepSeekExpertSlotState::kH2dInFlight)) ||
      (demand.disposition == DeepSeekExpertDemandDisposition::kResidentHit &&
       slot.state == DeepSeekExpertSlotState::kResident);
  if (!disposition_matches) {
    return Status::FailedPrecondition(
        "DeepSeek observed pager cost disposition drifted");
  }
  return ledger.debit(key, demand.disposition, observed_cost_ns);
}

}  // namespace pih
