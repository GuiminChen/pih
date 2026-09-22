#pragma once

#include <span>
#include <vector>

#include "pih/core/sha256.h"

namespace pih {

enum class EngineGpuProcessInventoryState : std::uint8_t {
  kUnknown,
  kOccupied,
  kEmpty,
};

struct EngineGpuProcessInventoryEntry final {
  std::uint64_t physical_gpu_identity = 0;
  std::vector<std::uint64_t> sorted_process_identities;
};

struct EngineGpuProcessInventoryReceipt final {
  std::uint64_t engine_generation = 0;
  Sha256Digest allocation_lease_digest{};
  std::uint64_t sample_identity = 0;
  std::uint64_t sample_started_ns = 0;
  std::uint64_t sample_completed_ns = 0;
  bool visibility_complete = false;
  std::vector<EngineGpuProcessInventoryEntry> devices;
};

class EngineGpuProcessInventoryGate final {
 public:
  static Result<EngineGpuProcessInventoryGate> Create(
      std::uint64_t engine_generation, Sha256Digest allocation_lease_digest,
      std::span<const std::uint64_t> sorted_physical_gpu_identities);
  Result<EngineGpuProcessInventoryState> accept(
      const EngineGpuProcessInventoryReceipt& receipt);
  [[nodiscard]] EngineGpuProcessInventoryState state() const noexcept {
    return state_;
  }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] bool bound_to(std::uint64_t generation,
                              const Sha256Digest& lease_digest) const noexcept;

 private:
  EngineGpuProcessInventoryGate(
      std::uint64_t generation, Sha256Digest lease_digest,
      std::vector<std::uint64_t> devices) noexcept
      : generation_(generation), lease_digest_(lease_digest),
        devices_(std::move(devices)) {}
  std::uint64_t generation_ = 0;
  Sha256Digest lease_digest_{};
  std::vector<std::uint64_t> devices_;
  std::uint64_t last_sample_identity_ = 0;
  std::uint64_t last_sample_completed_ns_ = 0;
  EngineGpuProcessInventoryState state_ =
      EngineGpuProcessInventoryState::kUnknown;
  bool poisoned_ = false;
};

}  // namespace pih
