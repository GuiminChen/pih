#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pih/io/controller_file_lease.h"
#include "pih/io/descriptor_mapped_range.h"
#include "pih/model/deepseek_stage_mapping_plan.h"
#include "pih/model/deepseek_verified_artifact_lease.h"

namespace pih {

class DeepSeekRankArtifactPrefaultTransaction;

struct DeepSeekWorkerShardDescriptor final {
  DeepSeekWorkerShardDescriptor(std::string name,
                                ArtifactWorkerDescriptor worker_descriptor)
      : shard_name(std::move(name)), descriptor(std::move(worker_descriptor)) {}
  DeepSeekWorkerShardDescriptor(const DeepSeekWorkerShardDescriptor&) = delete;
  DeepSeekWorkerShardDescriptor& operator=(
      const DeepSeekWorkerShardDescriptor&) = delete;
  DeepSeekWorkerShardDescriptor(DeepSeekWorkerShardDescriptor&&) noexcept =
      default;
  DeepSeekWorkerShardDescriptor& operator=(
      DeepSeekWorkerShardDescriptor&&) noexcept = default;

  std::string shard_name;
  ArtifactWorkerDescriptor descriptor;
};

struct DeepSeekCapabilityShardLease final {
  std::string shard_name;
  std::shared_ptr<DeepSeekVerifiedArtifactLease> lease;
};

class DeepSeekStageMappedInventory final {
 public:
  static Result<DeepSeekStageMappedInventory> Create(
      const DeepSeekRankMappingPlan& plan,
      std::vector<DeepSeekWorkerShardDescriptor> descriptors);
  static Result<DeepSeekStageMappedInventory> CreateCapabilityBacked(
      const DeepSeekRankMappingPlan& plan,
      std::vector<DeepSeekCapabilityShardLease> leases);

  DeepSeekStageMappedInventory(const DeepSeekStageMappedInventory&) = delete;
  DeepSeekStageMappedInventory& operator=(
      const DeepSeekStageMappedInventory&) = delete;
  DeepSeekStageMappedInventory(DeepSeekStageMappedInventory&&) noexcept =
      default;
  DeepSeekStageMappedInventory& operator=(
      DeepSeekStageMappedInventory&& other) noexcept;

  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint64_t mapped_interval_bytes() const noexcept {
    return mapped_interval_bytes_;
  }
  Result<std::span<const std::byte>> bytes(std::string_view shard_name,
                                          std::uint64_t file_offset,
                                          std::uint64_t bytes) const;

 private:
  friend class DeepSeekRankArtifactPrefaultTransaction;
  DeepSeekStageMappedInventory() = default;

  struct CapabilityMapping final {
    std::string shard_name;
    DeepSeekVerifiedArtifactMapping range;
  };

  std::uint32_t rank_ = 0;
  std::uint64_t mapped_interval_bytes_ = 0;
  std::vector<DeepSeekCapabilityShardLease> capability_leases_;
  std::vector<CapabilityMapping> capability_mappings_;
};

}  // namespace pih
