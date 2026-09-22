#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_mapped_tensor_source.h"
#include "pih/model/deepseek_rank_artifact_metadata_receipt.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankArtifactMappingOwnerAbi =
    "pih_deepseek_rank_artifact_mapping_owner_v1";

Result<Sha256Digest> compile_deepseek_rank_artifact_mapping_owner_root(
    const DeepSeekRankArtifactTransferManifest& transfer_manifest,
    const DeepSeekRankArtifactMetadataBlob& blob,
    const Sha256Digest& metadata_receipt_root,
    const Sha256Digest& descriptor_transaction_root,
    const Sha256Digest& metadata_transaction_root);

// Stable-address worker owner for read-only rank mappings. Creation is the
// sole consumer of a complete descriptor+metadata receipt. The object is
// neither copyable nor movable because its tensor source points at its mapped
// inventory member.
class DeepSeekRankArtifactMappingOwner final {
 public:
  static Result<std::unique_ptr<DeepSeekRankArtifactMappingOwner>> Create(
      DeepSeekRankArtifactMetadataReceipt receipt);

  DeepSeekRankArtifactMappingOwner(
      const DeepSeekRankArtifactMappingOwner&) = delete;
  DeepSeekRankArtifactMappingOwner& operator=(
      const DeepSeekRankArtifactMappingOwner&) = delete;
  DeepSeekRankArtifactMappingOwner(
      DeepSeekRankArtifactMappingOwner&&) = delete;
  DeepSeekRankArtifactMappingOwner& operator=(
      DeepSeekRankArtifactMappingOwner&&) = delete;

  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] std::size_t tensor_count() const noexcept {
    return tensor_source_ ? tensor_source_->size() : 0;
  }
  [[nodiscard]] std::uint64_t mapped_interval_bytes() const noexcept {
    return inventory_.mapped_interval_bytes();
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
  [[nodiscard]] const Sha256Digest& metadata_receipt_root() const noexcept {
    return metadata_receipt_root_;
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
  [[nodiscard]] const DeepSeekMappedTensorSource& tensor_source()
      const noexcept {
    return *tensor_source_;
  }
  [[nodiscard]] const DeepSeekStageMappedInventory& inventory()
      const noexcept {
    return inventory_;
  }
  [[nodiscard]] const DeepSeekRankMappingPlan& mapping_plan()
      const noexcept {
    return mapping_plan_;
  }
  [[nodiscard]] std::span<const DeepSeekRankTensorRecord> tensor_records()
      const noexcept {
    return tensor_records_;
  }
  Result<DeepSeekMappedTensorSpan> resolve(
      std::string_view tensor_name) const;

 private:
  DeepSeekRankArtifactMappingOwner(
      DeepSeekStageMappedInventory inventory, std::uint64_t engine_epoch,
      std::uint64_t worker_generation, std::uint32_t world_size,
      std::uint32_t rank,
      ArtifactImmutabilityMode immutability_mode,
      bool source_catalog_production_eligible, bool dspark_enabled,
      Sha256Digest metadata_receipt_root,
      Sha256Digest descriptor_transaction_root,
      Sha256Digest metadata_transaction_root, Sha256Digest metadata_root,
      Sha256Digest mapping_owner_root,
      DeepSeekRankMappingPlan mapping_plan,
      std::vector<DeepSeekRankTensorRecord> tensor_records) noexcept;

  // tensor_source_ must be destroyed before its retained records, mapping
  // plan and inventory, so it is declared after every possible pointee.
  // This object never moves after tensor_source_ is constructed.
  DeepSeekStageMappedInventory inventory_;
  DeepSeekRankMappingPlan mapping_plan_;
  std::vector<DeepSeekRankTensorRecord> tensor_records_;
  std::optional<DeepSeekMappedTensorSource> tensor_source_;
  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint32_t rank_ = 0;
  ArtifactImmutabilityMode immutability_mode_ =
      ArtifactImmutabilityMode::kUncalibrated;
  bool source_catalog_production_eligible_ = false;
  bool dspark_enabled_ = false;
  Sha256Digest metadata_receipt_root_{};
  Sha256Digest descriptor_transaction_root_{};
  Sha256Digest metadata_transaction_root_{};
  Sha256Digest metadata_root_{};
  Sha256Digest mapping_owner_root_{};
};

}  // namespace pih
