#pragma once

#include <span>
#include <array>
#include <vector>

#include "pih/core/sha256.h"

namespace pih {

enum class EngineOwnedResourceKind : std::uint8_t {
  kShm,
  kNetwork,
  kPinnedMemory,
  kListener,
  kArtifact,
  kGpuAllocation,
};

enum class EngineOwnedResourceBaselineState : std::uint8_t {
  kUnknown,
  kDrifted,
  kBaseline,
};

struct EngineOwnedResourceBaselineEntry final {
  EngineOwnedResourceKind kind = EngineOwnedResourceKind::kShm;
  Sha256Digest baseline_digest{};
};

struct EngineOwnedResourceObservation final {
  EngineOwnedResourceKind kind = EngineOwnedResourceKind::kShm;
  bool visibility_complete = false;
  Sha256Digest observed_digest{};
};

struct EngineOwnedResourceBaselineReceipt final {
  std::uint64_t engine_generation = 0;
  Sha256Digest allocation_lease_digest{};
  std::uint64_t sample_identity = 0;
  std::uint64_t sample_started_ns = 0;
  std::uint64_t sample_completed_ns = 0;
  std::vector<EngineOwnedResourceObservation> resources;
};

class EngineOwnedResourceBaselineGate final {
 public:
  static Result<EngineOwnedResourceBaselineGate> Create(
      std::uint64_t engine_generation, Sha256Digest allocation_lease_digest,
      std::span<const EngineOwnedResourceBaselineEntry> baselines);
  Result<EngineOwnedResourceBaselineState> accept(
      const EngineOwnedResourceBaselineReceipt& receipt);
  [[nodiscard]] EngineOwnedResourceBaselineState state() const noexcept {
    return state_;
  }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] bool bound_to(std::uint64_t generation,
                              const Sha256Digest& lease_digest) const noexcept;
  [[nodiscard]] bool visibility_complete(
      EngineOwnedResourceKind kind) const noexcept;
  [[nodiscard]] bool at_baseline(EngineOwnedResourceKind kind) const noexcept;

 private:
  EngineOwnedResourceBaselineGate(
      std::uint64_t generation, Sha256Digest lease_digest,
      std::vector<EngineOwnedResourceBaselineEntry> baselines) noexcept
      : generation_(generation), lease_digest_(lease_digest),
        baselines_(std::move(baselines)) {}
  std::uint64_t generation_ = 0;
  Sha256Digest lease_digest_{};
  std::vector<EngineOwnedResourceBaselineEntry> baselines_;
  std::uint64_t last_sample_identity_ = 0;
  std::uint64_t last_sample_completed_ns_ = 0;
  EngineOwnedResourceBaselineState state_ =
      EngineOwnedResourceBaselineState::kUnknown;
  std::array<bool, 6> visibility_{};
  std::array<bool, 6> at_baseline_{};
  bool poisoned_ = false;
};

}  // namespace pih
