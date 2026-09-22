#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pih/io/controller_file_lease.h"
#include "pih/model/deepseek_runtime_artifact_manifest.h"
#include "pih/model/deepseek_runtime_records_manifest.h"
#include "pih/model/deepseek_rank_tensor_record.h"
#include "pih/model/deepseek_stage_mapped_inventory.h"
#include "pih/model/deepseek_stage_mapping_plan.h"
#include "pih/model/safetensors_file.h"
#include "pih/model/safetensors_shard_index.h"

namespace pih {

struct DeepSeekShardFsVerityDigest final {
  std::string shard_name;
  Sha256Digest expected_digest;
};

struct DeepSeekTargetMemberFsVerityDigest final {
  std::string member_name;
  Sha256Digest expected_digest;
};

class DeepSeekControllerArtifactCatalog final {
 public:
  static constexpr std::size_t kMaximumTargetMemberCount =
      DeepSeekRuntimeArtifactManifest::kMaximumTargetMemberCount;

  // Legacy source-repository paths retained for offline compiler tests. They
  // do not set target_authority_bound and NVIDIA bootstrap rejects them.
  static Result<DeepSeekControllerArtifactCatalog> Open(
      const std::filesystem::path& trusted_root,
      const SafetensorsShardIndex& index,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes,
      ArtifactImmutabilityMode immutability_mode);
  static Result<DeepSeekControllerArtifactCatalog> OpenFlash0731(
      const std::filesystem::path& trusted_root,
      const SafetensorsShardIndex& index,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes,
      ArtifactImmutabilityMode immutability_mode);
  static Result<DeepSeekControllerArtifactCatalog> OpenFlash0731FsVerity(
      const std::filesystem::path& trusted_root,
      const SafetensorsShardIndex& index,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes,
      std::span<const DeepSeekShardFsVerityDigest> expected_digests);
  static Result<DeepSeekControllerArtifactCatalog> OpenFlash0731DmVerity(
      const std::filesystem::path& trusted_root,
      const SafetensorsShardIndex& index,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes,
      const DmVeritySupervisorReceipt& supervisor_receipt,
      const Sha256Digest& expected_root_digest,
      const Sha256Digest& expected_table_digest,
      const Sha256Digest& expected_supervisor_attestation_digest,
      std::uint64_t integrity_reserve_bytes);
  // Root-authoritative target-generation path. The development entry point
  // accepts reduced fixtures; the Flash-0731 entry point additionally enforces
  // the exact family tensor/shard geometry.
  static Result<DeepSeekControllerArtifactCatalog> OpenTargetGeneration(
      const std::filesystem::path& trusted_root,
      const Sha256Digest& expected_artifact_root,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes,
      ArtifactImmutabilityMode immutability_mode);
  static Result<DeepSeekControllerArtifactCatalog>
  OpenFlash0731TargetGeneration(
      const std::filesystem::path& trusted_root,
      const Sha256Digest& expected_artifact_root,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes,
      ArtifactImmutabilityMode immutability_mode);
  static Result<DeepSeekControllerArtifactCatalog>
  OpenFlash0731TargetGenerationFsVerity(
      const std::filesystem::path& trusted_root,
      const Sha256Digest& expected_artifact_root,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes,
      std::span<const DeepSeekTargetMemberFsVerityDigest> expected_digests);

  DeepSeekControllerArtifactCatalog(
      const DeepSeekControllerArtifactCatalog&) = delete;
  DeepSeekControllerArtifactCatalog& operator=(
      const DeepSeekControllerArtifactCatalog&) = delete;
  DeepSeekControllerArtifactCatalog(
      DeepSeekControllerArtifactCatalog&&) noexcept = default;
  DeepSeekControllerArtifactCatalog& operator=(
      DeepSeekControllerArtifactCatalog&&) noexcept = default;

  [[nodiscard]] std::size_t master_shard_count() const noexcept {
    return shards_.size();
  }
  [[nodiscard]] const DeepSeekStageMappingPlan& mapping_plan() const noexcept {
    return mapping_plan_;
  }
  [[nodiscard]] ArtifactImmutabilityMode immutability_mode() const noexcept {
    return immutability_mode_;
  }
  [[nodiscard]] std::uint64_t integrity_owner_bytes() const noexcept {
    return integrity_owner_bytes_;
  }
  [[nodiscard]] bool target_authority_bound() const noexcept {
    return target_authority_bound_;
  }
  [[nodiscard]] bool production_eligible() const noexcept {
    return production_eligible_;
  }
  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_root_;
  }
  [[nodiscard]] const SafetensorsHeader* header(
      std::string_view shard_name) const noexcept;
  Result<Sha256Digest> shard_enforced_digest(
      std::string_view shard_name) const;
  Result<std::vector<DeepSeekWorkerShardDescriptor>>
  duplicate_rank_descriptors(std::uint32_t rank) const;
  Result<std::span<const DeepSeekRankTensorRecord>> rank_tensor_records(
      std::uint32_t rank) const;
  Status poll_master_leases(
      const DmVeritySupervisorReceipt* current_dm_receipt = nullptr) const;

 private:
  static Result<DeepSeekControllerArtifactCatalog> OpenImpl(
      const std::filesystem::path& trusted_root,
      const SafetensorsShardIndex& index,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes,
      ArtifactImmutabilityMode immutability_mode,
      std::span<const DeepSeekShardFsVerityDigest> expected_digests,
      const DmVeritySupervisorReceipt* dm_receipt = nullptr,
      const Sha256Digest* expected_dm_root = nullptr,
      const Sha256Digest* expected_dm_table = nullptr,
      const Sha256Digest* expected_dm_attestation = nullptr,
      std::uint64_t integrity_reserve_bytes = 0);
  static Result<DeepSeekControllerArtifactCatalog> OpenTargetImpl(
      const std::filesystem::path& trusted_root,
      const Sha256Digest& expected_artifact_root,
      const DeepSeekPipelinePlan& pipeline,
      std::uint64_t maximum_shard_bytes,
      ArtifactImmutabilityMode immutability_mode,
      bool require_flash_0731_geometry,
      std::span<const DeepSeekTargetMemberFsVerityDigest> expected_digests);

  struct Shard final {
    Shard(std::string shard_name, ControllerFileLease master_lease,
          SafetensorsHeaderFileReceipt header_receipt)
        : name(std::move(shard_name)),
          lease(std::move(master_lease)),
          receipt(std::move(header_receipt)) {}
    Shard(Shard&&) noexcept = default;
    Shard& operator=(Shard&&) noexcept = default;
    Shard(const Shard&) = delete;
    Shard& operator=(const Shard&) = delete;

    std::string name;
    ControllerFileLease lease;
    SafetensorsHeaderFileReceipt receipt;
  };

  struct AuthorityFile final {
    AuthorityFile(std::string member_name, ControllerFileLease master_lease)
        : name(std::move(member_name)), lease(std::move(master_lease)) {}
    AuthorityFile(AuthorityFile&&) noexcept = default;
    AuthorityFile& operator=(AuthorityFile&&) noexcept = default;
    AuthorityFile(const AuthorityFile&) = delete;
    AuthorityFile& operator=(const AuthorityFile&) = delete;

    std::string name;
    ControllerFileLease lease;
  };

  DeepSeekControllerArtifactCatalog() = default;

  std::vector<Shard> shards_;
  std::vector<AuthorityFile> authority_files_;
  DeepSeekStageMappingPlan mapping_plan_;
  std::vector<std::vector<DeepSeekRankTensorRecord>> rank_tensors_;
  ArtifactImmutabilityMode immutability_mode_ =
      ArtifactImmutabilityMode::kUncalibrated;
  std::uint64_t integrity_owner_bytes_ = 0;
  bool target_authority_bound_ = false;
  bool production_eligible_ = false;
  Sha256Digest artifact_root_;
};

}  // namespace pih
