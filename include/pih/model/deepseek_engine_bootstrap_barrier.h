#pragma once

#include <cstdint>
#include <vector>

#include "pih/model/deepseek_artifact_epoch_guard.h"
#include "pih/model/deepseek_expert_pager.h"
#include "pih/model/deepseek_stage_mapped_inventory.h"
#include "pih/model/deepseek_weight_finalizer.h"

namespace pih {

class DeepSeekRankBootstrapReceipt final {
 public:
  static Result<DeepSeekRankBootstrapReceipt> Create(
      std::uint64_t epoch, std::uint32_t world_size,
      const DeepSeekStageMappedInventory& inventory,
      const DeepSeekWeightMaterializationPlan& materialization,
      const DeepSeekWeightFinalizer& finalizer,
      DeepSeekRoutedExpertResidency expert_residency,
      const DeepSeekExpertPager* pager);

  [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint32_t world_size() const noexcept { return world_size_; }
  [[nodiscard]] std::uint64_t mapped_bytes() const noexcept {
    return mapped_bytes_;
  }
  [[nodiscard]] const Sha256Digest& weight_layout_digest() const noexcept {
    return weight_layout_digest_;
  }
  [[nodiscard]] const Sha256Digest& weight_seal_digest() const noexcept {
    return weight_seal_digest_;
  }

 private:
  std::uint64_t epoch_ = 0;
  std::uint32_t rank_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint64_t mapped_bytes_ = 0;
  std::uint64_t fixed_backing_bytes_ = 0;
  std::uint64_t paged_source_bytes_ = 0;
  std::uint32_t expert_slot_count_ = 0;
  std::uint32_t staging_extent_count_ = 0;
  Sha256Digest weight_layout_digest_{};
  Sha256Digest weight_seal_digest_{};
};

class DeepSeekEngineBootstrapBarrier final {
 public:
  static Result<DeepSeekEngineBootstrapBarrier> Create(
      DeepSeekArtifactEpochGuard& artifact_guard);

  Status accept(const DeepSeekRankBootstrapReceipt& receipt);
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::uint32_t accepted_rank_count() const noexcept {
    return accepted_rank_count_;
  }

 private:
  explicit DeepSeekEngineBootstrapBarrier(
      DeepSeekArtifactEpochGuard& artifact_guard)
      : artifact_guard_(&artifact_guard),
        accepted_(artifact_guard.world_size(), false) {}

  DeepSeekArtifactEpochGuard* artifact_guard_ = nullptr;
  std::vector<bool> accepted_;
  std::uint32_t accepted_rank_count_ = 0;
};

}  // namespace pih
