#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "pih/model/deepseek_rank_artifact_metadata_transfer_transaction.h"
#include "pih/model/deepseek_rank_post_exec_resource_receipt.h"

namespace pih {

inline constexpr std::string_view
    kDeepSeekRankPostMappingResourceReceiptAbi =
        "pih_deepseek_rank_post_mapping_resource_receipt_v1";
inline constexpr std::string_view kDeepSeekRankPostMappingResourceSealAbi =
    "pih_deepseek_rank_post_mapping_resource_seal_v1";

// Worker observation taken only after the stable mapping owner exists. The
// embedded resource observation reuses the exact OS counter schema but is
// bound to a distinct post-mapping proof domain below.
struct DeepSeekRankPostMappingResourceObservation final {
  DeepSeekRankPostExecResourceObservation resources;
  Sha256Digest first_resource_seal_root{};
  Sha256Digest descriptor_transaction_root{};
  Sha256Digest metadata_transaction_root{};
  Sha256Digest metadata_root{};
  Sha256Digest mapping_owner_root{};
  std::uint64_t mapped_interval_bytes = 0;
  ArtifactImmutabilityMode immutability_mode =
      ArtifactImmutabilityMode::kUncalibrated;
  bool source_catalog_production_eligible = false;
  bool dspark_enabled = false;
};

class DeepSeekRankPostMappingResourceReceipt final {
 public:
  static Result<DeepSeekRankPostMappingResourceReceipt> Compile(
      const DeepSeekRankProcessSupervisor& supervisor,
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankSpawnResourcePlan& spawn_plan,
      std::span<const DeepSeekRankPostExecResourcePlan> resource_plans,
      const DeepSeekRankPostExecResourceSeal& first_resource_seal,
      const DeepSeekRankArtifactMetadataTransferTransaction&
          metadata_transaction,
      const DeepSeekRankPostMappingResourceObservation& observation);

  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint64_t process_identity() const noexcept {
    return process_identity_;
  }
  [[nodiscard]] const Sha256Digest& capacity_plan_instance_root()
      const noexcept {
    return capacity_plan_instance_root_;
  }
  [[nodiscard]] const Sha256Digest& os_resource_envelope_root()
      const noexcept {
    return os_resource_envelope_root_;
  }
  [[nodiscard]] const Sha256Digest& first_resource_seal_root()
      const noexcept {
    return first_resource_seal_root_;
  }
  [[nodiscard]] const Sha256Digest& resource_plan_root() const noexcept {
    return resource_plan_root_;
  }
  [[nodiscard]] const Sha256Digest& descriptor_transaction_root()
      const noexcept {
    return descriptor_transaction_root_;
  }
  [[nodiscard]] const Sha256Digest& metadata_transaction_root()
      const noexcept {
    return metadata_transaction_root_;
  }
  [[nodiscard]] const Sha256Digest& metadata_root() const noexcept {
    return metadata_root_;
  }
  [[nodiscard]] const Sha256Digest& mapping_owner_root() const noexcept {
    return mapping_owner_root_;
  }
  [[nodiscard]] std::uint64_t mapped_interval_bytes() const noexcept {
    return mapped_interval_bytes_;
  }
  [[nodiscard]] ArtifactImmutabilityMode immutability_mode() const noexcept {
    return immutability_mode_;
  }
  [[nodiscard]] bool source_catalog_production_eligible() const noexcept {
    return source_catalog_production_eligible_;
  }
  [[nodiscard]] bool dspark_enabled() const noexcept {
    return dspark_enabled_;
  }
  [[nodiscard]] const Sha256Digest& receipt_root() const noexcept {
    return receipt_root_;
  }

 private:
  DeepSeekRankPostMappingResourceReceipt(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size, std::uint32_t rank,
      std::uint64_t process_identity,
      Sha256Digest capacity_plan_instance_root,
      Sha256Digest os_resource_envelope_root,
      Sha256Digest first_resource_seal_root,
      Sha256Digest resource_plan_root,
      Sha256Digest descriptor_transaction_root,
      Sha256Digest metadata_transaction_root, Sha256Digest metadata_root,
      Sha256Digest mapping_owner_root,
      std::uint64_t mapped_interval_bytes,
      ArtifactImmutabilityMode immutability_mode,
      bool source_catalog_production_eligible, bool dspark_enabled,
      Sha256Digest receipt_root) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint32_t rank_ = 0;
  std::uint64_t process_identity_ = 0;
  Sha256Digest capacity_plan_instance_root_{};
  Sha256Digest os_resource_envelope_root_{};
  Sha256Digest first_resource_seal_root_{};
  Sha256Digest resource_plan_root_{};
  Sha256Digest descriptor_transaction_root_{};
  Sha256Digest metadata_transaction_root_{};
  Sha256Digest metadata_root_{};
  Sha256Digest mapping_owner_root_{};
  std::uint64_t mapped_interval_bytes_ = 0;
  ArtifactImmutabilityMode immutability_mode_ =
      ArtifactImmutabilityMode::kUncalibrated;
  bool source_catalog_production_eligible_ = false;
  bool dspark_enabled_ = false;
  Sha256Digest receipt_root_{};
};

class DeepSeekRankPostMappingResourceSeal final {
 public:
  static Result<DeepSeekRankPostMappingResourceSeal> Compile(
      const DeepSeekRankProcessSupervisor& supervisor,
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankSpawnResourcePlan& spawn_plan,
      std::span<const DeepSeekRankPostExecResourcePlan> resource_plans,
      const DeepSeekRankPostExecResourceSeal& first_resource_seal,
      const DeepSeekRankArtifactMetadataTransferTransaction&
          metadata_transaction,
      std::span<const DeepSeekRankPostMappingResourceReceipt> receipts);

  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] const Sha256Digest& first_resource_seal_root()
      const noexcept {
    return first_resource_seal_root_;
  }
  [[nodiscard]] const Sha256Digest& metadata_transaction_root()
      const noexcept {
    return metadata_transaction_root_;
  }
  [[nodiscard]] const Sha256Digest& mapping_owner_set_root()
      const noexcept {
    return mapping_owner_set_root_;
  }
  [[nodiscard]] const Sha256Digest& receipt_set_root() const noexcept {
    return receipt_set_root_;
  }
  [[nodiscard]] bool production_eligible() const noexcept {
    return production_eligible_;
  }
  [[nodiscard]] bool dspark_enabled() const noexcept {
    return dspark_enabled_;
  }
  [[nodiscard]] const Sha256Digest& seal_root() const noexcept {
    return seal_root_;
  }

 private:
  DeepSeekRankPostMappingResourceSeal(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size, Sha256Digest first_resource_seal_root,
      Sha256Digest metadata_transaction_root,
      Sha256Digest mapping_owner_set_root,
      Sha256Digest receipt_set_root, bool production_eligible,
      bool dspark_enabled, Sha256Digest seal_root) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  Sha256Digest first_resource_seal_root_{};
  Sha256Digest metadata_transaction_root_{};
  Sha256Digest mapping_owner_set_root_{};
  Sha256Digest receipt_set_root_{};
  bool production_eligible_ = false;
  bool dspark_enabled_ = false;
  Sha256Digest seal_root_{};
};

}  // namespace pih
