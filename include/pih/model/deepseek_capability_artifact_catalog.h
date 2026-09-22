#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/contracts/verified_artifact_v1.h"
#include "pih/model/deepseek_rank_tensor_record.h"
#include "pih/model/deepseek_stage_mapped_inventory.h"
#include "pih/model/deepseek_stage_mapping_plan.h"
#include "pih/model/deepseek_verified_artifact_lease.h"

namespace pih {

// Native-plugin PP1 catalog. All artifact access stays behind the storage
// plugin ABI; no controller filesystem descriptor is opened or duplicated.
class DeepSeekCapabilityArtifactCatalog final {
 public:
  static Result<DeepSeekCapabilityArtifactCatalog>
  OpenFlash0731TargetGeneration(
      const pih_verified_artifact_api_v1& artifact_api,
      const std::filesystem::path& trusted_root,
      const Sha256Digest& expected_artifact_root,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes);

  DeepSeekCapabilityArtifactCatalog(
      const DeepSeekCapabilityArtifactCatalog&) = delete;
  DeepSeekCapabilityArtifactCatalog& operator=(
      const DeepSeekCapabilityArtifactCatalog&) = delete;
  DeepSeekCapabilityArtifactCatalog(
      DeepSeekCapabilityArtifactCatalog&&) noexcept = default;
  DeepSeekCapabilityArtifactCatalog& operator=(
      DeepSeekCapabilityArtifactCatalog&&) noexcept = default;

  [[nodiscard]] const DeepSeekStageMappingPlan& mapping_plan() const noexcept {
    return mapping_plan_;
  }
  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_root_;
  }
  Result<std::vector<DeepSeekCapabilityShardLease>> rank_leases(
      std::uint32_t rank) const;
  Result<std::span<const DeepSeekRankTensorRecord>> rank_tensor_records(
      std::uint32_t rank) const;
  Status poll_leases() const;

 private:
  struct Shard final {
    std::string name;
    std::shared_ptr<DeepSeekVerifiedArtifactLease> lease;
    SafetensorsHeaderFileReceipt receipt;
  };

  DeepSeekCapabilityArtifactCatalog() = default;
  [[nodiscard]] const SafetensorsHeader* header(
      std::string_view shard_name) const noexcept;

  std::vector<Shard> shards_;
  std::vector<std::shared_ptr<DeepSeekVerifiedArtifactLease>>
      authority_leases_;
  DeepSeekStageMappingPlan mapping_plan_;
  std::vector<std::vector<DeepSeekRankTensorRecord>> rank_tensors_;
  Sha256Digest artifact_root_;
};

}  // namespace pih
