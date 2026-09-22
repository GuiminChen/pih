#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_tensor_record.h"
#include "pih/model/deepseek_runtime_artifact_manifest.h"
#include "pih/model/deepseek_stage_mapped_inventory.h"

namespace pih {

class DeepSeekControllerArtifactCatalog;
struct DeepSeekRankArtifactDescriptorExpectation;

inline constexpr std::string_view kDeepSeekRankArtifactHandoffPlanAbi =
    "pih_deepseek_rank_artifact_handoff_plan_v1";

Result<Sha256Digest> compile_deepseek_stage_mapping_plan_root(
    const DeepSeekStageMappingPlan& mapping);
Result<Sha256Digest> compile_deepseek_rank_mapping_plan_root(
    const DeepSeekRankMappingPlan& mapping);
Result<Sha256Digest> compile_deepseek_rank_tensor_handoff_root(
    std::span<const DeepSeekRankTensorRecord> records);
Result<Sha256Digest> compile_deepseek_rank_artifact_descriptor_handoff_root(
    std::span<const DeepSeekRankArtifactDescriptorExpectation>
        descriptors);
Result<Sha256Digest> compile_deepseek_rank_artifact_handoff_manifest_root(
    std::uint32_t rank, const DeepSeekRankMappingPlan& mapping,
    std::span<const DeepSeekRankArtifactDescriptorExpectation> descriptors,
    std::span<const DeepSeekRankTensorRecord> tensor_records);
Result<Sha256Digest>
compile_deepseek_rank_artifact_handoff_manifest_root_from_roots(
    std::uint32_t rank, const Sha256Digest& rank_mapping_root,
    const Sha256Digest& descriptor_root, const Sha256Digest& tensor_root,
    std::uint32_t descriptor_count, std::uint32_t tensor_record_count);

struct DeepSeekRankArtifactDescriptorExpectation final {
  std::uint32_t ordinal = 0;
  std::string shard_name;
  ArtifactFileIdentity identity;
  ArtifactImmutabilityMode immutability_mode =
      ArtifactImmutabilityMode::kUncalibrated;
  Sha256Digest enforced_digest{};
};

class DeepSeekRankArtifactHandoffManifest final {
 public:
  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] const DeepSeekRankMappingPlan& mapping() const noexcept {
    return mapping_;
  }
  [[nodiscard]] std::span<const DeepSeekRankTensorRecord> tensor_records()
      const noexcept {
    return tensor_records_;
  }
  [[nodiscard]] std::span<const DeepSeekRankArtifactDescriptorExpectation>
  descriptor_expectations() const noexcept {
    return descriptor_expectations_;
  }
  [[nodiscard]] const Sha256Digest& manifest_root() const noexcept {
    return manifest_root_;
  }

 private:
  friend class DeepSeekRankArtifactHandoffPlan;
  DeepSeekRankArtifactHandoffManifest(
      std::uint32_t rank, DeepSeekRankMappingPlan mapping,
      std::vector<DeepSeekRankTensorRecord> tensor_records,
      std::vector<DeepSeekRankArtifactDescriptorExpectation>
          descriptor_expectations,
      Sha256Digest manifest_root) noexcept;

  std::uint32_t rank_ = 0;
  DeepSeekRankMappingPlan mapping_;
  std::vector<DeepSeekRankTensorRecord> tensor_records_;
  std::vector<DeepSeekRankArtifactDescriptorExpectation>
      descriptor_expectations_;
  Sha256Digest manifest_root_{};
};

// Controller-side descriptor inventory prepared for a later acknowledged
// SCM_RIGHTS transaction. It owns worker descriptor duplicates, but exposes no
// send/release operation and therefore cannot by itself authorize model load.
class DeepSeekRankArtifactHandoffPlan final {
 public:
  static Result<DeepSeekRankArtifactHandoffPlan> Compile(
      const DeepSeekControllerArtifactCatalog& catalog);

  DeepSeekRankArtifactHandoffPlan(
      const DeepSeekRankArtifactHandoffPlan&) = delete;
  DeepSeekRankArtifactHandoffPlan& operator=(
      const DeepSeekRankArtifactHandoffPlan&) = delete;
  DeepSeekRankArtifactHandoffPlan(
      DeepSeekRankArtifactHandoffPlan&&) noexcept = default;
  DeepSeekRankArtifactHandoffPlan& operator=(
      DeepSeekRankArtifactHandoffPlan&&) noexcept = default;

  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] ArtifactImmutabilityMode immutability_mode() const noexcept {
    return immutability_mode_;
  }
  [[nodiscard]] bool source_catalog_production_eligible() const noexcept {
    return source_catalog_production_eligible_;
  }
  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_root_;
  }
  [[nodiscard]] const Sha256Digest& mapping_root() const noexcept {
    return mapping_root_;
  }
  [[nodiscard]] const Sha256Digest& plan_root() const noexcept {
    return plan_root_;
  }
  [[nodiscard]] const Sha256Digest& rank_root(std::uint32_t rank) const;
  [[nodiscard]] const DeepSeekRankArtifactHandoffManifest& rank_manifest(
      std::uint32_t rank) const;
  [[nodiscard]] std::span<const DeepSeekWorkerShardDescriptor> descriptors(
      std::uint32_t rank) const;

 private:
  DeepSeekRankArtifactHandoffPlan(
      std::uint32_t world_size, ArtifactImmutabilityMode immutability_mode,
      bool source_catalog_production_eligible, Sha256Digest artifact_root,
      Sha256Digest mapping_root, Sha256Digest plan_root,
      std::vector<DeepSeekRankArtifactHandoffManifest> rank_manifests,
      std::vector<std::vector<DeepSeekWorkerShardDescriptor>>
          descriptors) noexcept;

  std::uint32_t world_size_ = 0;
  ArtifactImmutabilityMode immutability_mode_ =
      ArtifactImmutabilityMode::kUncalibrated;
  bool source_catalog_production_eligible_ = false;
  Sha256Digest artifact_root_{};
  Sha256Digest mapping_root_{};
  Sha256Digest plan_root_{};
  std::vector<DeepSeekRankArtifactHandoffManifest> rank_manifests_;
  std::vector<std::vector<DeepSeekWorkerShardDescriptor>> descriptors_;
};

}  // namespace pih
