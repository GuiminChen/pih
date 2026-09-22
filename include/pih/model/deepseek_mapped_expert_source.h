#pragma once

#include <array>
#include <mutex>
#include <vector>

#include "pih/model/deepseek_expert_bundle_layout.h"
#include "pih/model/deepseek_expert_transfer_driver.h"
#include "pih/model/deepseek_stage_mapped_inventory.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

struct DeepSeekExpertMappedSegment final {
  std::string shard_name;
  std::uint64_t file_offset = 0;
  std::uint64_t bytes = 0;
};

struct DeepSeekExpertMappedBundleRecord final {
  DeepSeekExpertIdentity identity;
  std::array<DeepSeekExpertMappedSegment, 6> segments;
};

class DeepSeekMappedExpertSource final : public DeepSeekExpertHostSource {
 public:
  static Result<DeepSeekMappedExpertSource> Create(
      const DeepSeekStageMappedInventory& inventory,
      DeepSeekStageRange owned_layers,
      std::vector<DeepSeekExpertMappedBundleRecord> records,
      std::vector<DeepSeekPinnedExpertExtent> staging_extents,
      std::uint32_t transfer_reservation_window =
          DeepSeekExpertPager::kTransferReservationWindow);

  DeepSeekMappedExpertSource(const DeepSeekMappedExpertSource&) = delete;
  DeepSeekMappedExpertSource& operator=(const DeepSeekMappedExpertSource&) = delete;
  DeepSeekMappedExpertSource(DeepSeekMappedExpertSource&& other) noexcept;
  DeepSeekMappedExpertSource& operator=(DeepSeekMappedExpertSource&&) = delete;

  Result<DeepSeekPinnedExpertExtent> resolve(
      DeepSeekExpertIdentity identity, std::uint64_t bytes) override;
  Status release(DeepSeekExpertIdentity identity,
                 DeepSeekPinnedExpertExtent extent) override;

  [[nodiscard]] std::uint32_t staging_extent_count() const noexcept {
    return static_cast<std::uint32_t>(extents_.size());
  }

 private:
  struct ExtentState final {
    DeepSeekPinnedExpertExtent extent;
    bool leased = false;
    DeepSeekExpertIdentity identity{};
  };

  DeepSeekMappedExpertSource(const DeepSeekStageMappedInventory& inventory,
                             DeepSeekStageRange owned_layers,
                             std::vector<DeepSeekExpertMappedBundleRecord> records,
                             std::vector<ExtentState> extents)
      : inventory_(&inventory), owned_layers_(owned_layers),
        records_(std::move(records)), extents_(std::move(extents)) {}

  const DeepSeekStageMappedInventory* inventory_ = nullptr;
  DeepSeekStageRange owned_layers_;
  std::vector<DeepSeekExpertMappedBundleRecord> records_;
  std::vector<ExtentState> extents_;
  std::mutex mutex_;
};

}  // namespace pih
