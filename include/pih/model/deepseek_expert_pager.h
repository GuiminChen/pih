#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pih/model/deepseek_v4_config.h"

namespace pih {

class DeepSeekPagerCostLedger;
struct DeepSeekPagerCostKey;

struct DeepSeekExpertIdentity final {
  std::uint16_t layer = 0;
  std::uint16_t expert = 0;
  bool operator==(const DeepSeekExpertIdentity&) const = default;
};

enum class DeepSeekExpertSlotState : std::uint8_t {
  kAbsent,
  kReserved,
  kH2dInFlight,
  kResident,
  kEvictPending,
  kFailed,
};

enum class DeepSeekExpertDemandDisposition : std::uint8_t {
  kResidentHit,
  kInflightJoin,
  kNewMiss,
};

struct DeepSeekExpertDemand final {
  DeepSeekExpertDemandDisposition disposition;
  std::uint32_t slot;
  std::uint64_t generation;
};

struct DeepSeekExpertLease final {
  DeepSeekExpertIdentity identity;
  std::uint32_t slot;
  std::uint64_t generation;
};

class DeepSeekExpertPager final {
 public:
  static constexpr std::uint64_t kBundleBytes = 13369344ULL;
  static constexpr std::uint32_t kTransferReservationWindow = 2;

  static Result<DeepSeekExpertPager> Create(DeepSeekStageRange owned_layers,
                                            std::uint32_t slots,
                                            std::uint32_t staging_extents,
                                            std::uint32_t transfer_reservation_window =
                                                kTransferReservationWindow);
  Result<DeepSeekExpertDemand> demand(DeepSeekExpertIdentity identity);
  Status begin_h2d(DeepSeekExpertIdentity identity, std::uint64_t generation,
                   std::uint64_t payload_bytes);
  Status complete_h2d(DeepSeekExpertIdentity identity,
                      std::uint64_t generation);
  Result<DeepSeekExpertLease> acquire(DeepSeekExpertIdentity identity,
                                      std::uint64_t generation);
  Status release(const DeepSeekExpertLease& lease);
  Status request_eviction(DeepSeekExpertIdentity identity,
                          std::uint64_t generation);
  Status fail_transfer(DeepSeekExpertIdentity identity,
                       std::uint64_t generation);
  Result<std::uint64_t> debit_observed_cost(
      DeepSeekPagerCostLedger& ledger, DeepSeekPagerCostKey key,
      const DeepSeekExpertDemand& demand, std::uint64_t observed_cost_ns);
  void poison_epoch() noexcept { poisoned_ = true; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::uint32_t transfer_reservations() const noexcept {
    return transfer_reservations_;
  }
  [[nodiscard]] std::uint32_t slot_count() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
  }
  [[nodiscard]] std::uint32_t staging_extent_count() const noexcept {
    return staging_extents_;
  }
  [[nodiscard]] std::uint32_t transfer_reservation_window() const noexcept {
    return transfer_reservation_window_;
  }
  [[nodiscard]] DeepSeekStageRange owned_layers() const noexcept {
    return owned_layers_;
  }

 private:
  struct Slot final {
    DeepSeekExpertSlotState state = DeepSeekExpertSlotState::kAbsent;
    std::optional<DeepSeekExpertIdentity> identity;
    std::uint64_t generation = 0;
    std::uint32_t leases = 0;
  };
  DeepSeekStageRange owned_layers_;
  std::vector<Slot> slots_;
  std::uint32_t staging_extents_ = 0;
  std::uint32_t transfer_reservation_window_ = 0;
  std::uint32_t transfer_reservations_ = 0;
  std::uint64_t next_generation_ = 1;
  bool poisoned_ = false;
};

}  // namespace pih
