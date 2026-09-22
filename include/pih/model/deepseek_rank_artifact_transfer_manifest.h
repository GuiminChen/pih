#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "pih/io/controller_file_lease.h"
#include "pih/model/deepseek_runtime_records_manifest.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankArtifactTransferManifestAbi =
    "pih_deepseek_rank_artifact_transfer_manifest_v1";
inline constexpr std::string_view
    kDeepSeekRankArtifactTransferManifestFrameAbi =
        "pih_deepseek_rank_artifact_transfer_manifest_frame_v1";
inline constexpr std::size_t kDeepSeekRankArtifactTransferManifestBytes = 392;
inline constexpr std::uint32_t
    kDeepSeekRankArtifactTransferDescriptorBatchMaximum = 16;
inline constexpr std::uint32_t
    kDeepSeekRankArtifactTransferDescriptorMaximum = 50;

struct DeepSeekRankArtifactTransferManifestFields final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  std::uint64_t deadline_ns = 0;
  ArtifactImmutabilityMode immutability_mode =
      ArtifactImmutabilityMode::kUncalibrated;
  bool source_catalog_production_eligible = false;
  std::uint32_t descriptor_count = 0;
  std::uint32_t tensor_record_count = 0;
  std::uint32_t descriptor_batch_maximum =
      kDeepSeekRankArtifactTransferDescriptorBatchMaximum;
  std::uint32_t descriptor_batch_count = 0;
  Sha256Digest model_startup_plan_root{};
  Sha256Digest model_startup_rank_seed_root{};
  Sha256Digest capacity_plan_instance_root{};
  Sha256Digest artifact_admission_binding_root{};
  Sha256Digest artifact_handoff_plan_root{};
  Sha256Digest artifact_handoff_rank_root{};
  Sha256Digest artifact_root{};
  Sha256Digest mapping_root{};
};

class DeepSeekRankArtifactTransferManifest final {
 public:
  static Result<DeepSeekRankArtifactTransferManifest> Create(
      DeepSeekRankArtifactTransferManifestFields fields);

  [[nodiscard]] const DeepSeekRankArtifactTransferManifestFields& fields()
      const noexcept {
    return fields_;
  }
  [[nodiscard]] const Sha256Digest& manifest_root() const noexcept {
    return manifest_root_;
  }

 private:
  DeepSeekRankArtifactTransferManifest(
      DeepSeekRankArtifactTransferManifestFields fields,
      Sha256Digest manifest_root) noexcept;

  DeepSeekRankArtifactTransferManifestFields fields_;
  Sha256Digest manifest_root_{};
};

std::array<std::byte, kDeepSeekRankArtifactTransferManifestBytes>
encode_deepseek_rank_artifact_transfer_manifest(
    const DeepSeekRankArtifactTransferManifest& manifest);
Result<DeepSeekRankArtifactTransferManifest>
decode_deepseek_rank_artifact_transfer_manifest(
    std::span<const std::byte> frame);

}  // namespace pih
