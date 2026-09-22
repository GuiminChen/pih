#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_handoff_plan.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankArtifactMetadataBlobAbi =
    "pih_deepseek_rank_artifact_metadata_blob_v1";
inline constexpr std::string_view kDeepSeekRankArtifactMetadataBlobFrameAbi =
    "pih_deepseek_rank_artifact_metadata_blob_frame_v1";
inline constexpr std::size_t kDeepSeekRankArtifactMetadataBlobMaximumBytes =
    128ULL * 1024ULL * 1024ULL;

struct DeepSeekRankArtifactMetadataBlobFields final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  bool dspark_enabled = false;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  std::uint32_t descriptor_count = 0;
  Sha256Digest transfer_manifest_root{};
  Sha256Digest artifact_admission_binding_root{};
  Sha256Digest descriptor_transfer_transaction_root{};
  Sha256Digest artifact_handoff_rank_root{};
  Sha256Digest artifact_root{};
  Sha256Digest mapping_root{};
  Sha256Digest rank_mapping_root{};
  Sha256Digest descriptor_handoff_root{};
  Sha256Digest tensor_handoff_root{};
};

// Canonical rank-local mapping/tensor object. It contains no native handles;
// the later metadata receiver must join it to the descriptor adoption receipt.
class DeepSeekRankArtifactMetadataBlob final {
 public:
  static Result<DeepSeekRankArtifactMetadataBlob> Create(
      DeepSeekRankArtifactMetadataBlobFields fields,
      DeepSeekRankMappingPlan mapping,
      std::vector<DeepSeekRankTensorRecord> tensor_records);

  [[nodiscard]] const DeepSeekRankArtifactMetadataBlobFields& fields()
      const noexcept {
    return fields_;
  }
  [[nodiscard]] const DeepSeekRankMappingPlan& mapping() const noexcept {
    return mapping_;
  }
  [[nodiscard]] std::span<const DeepSeekRankTensorRecord> tensor_records()
      const noexcept {
    return tensor_records_;
  }
  [[nodiscard]] const Sha256Digest& metadata_root() const noexcept {
    return metadata_root_;
  }

 private:
  DeepSeekRankArtifactMetadataBlob(
      DeepSeekRankArtifactMetadataBlobFields fields,
      DeepSeekRankMappingPlan mapping,
      std::vector<DeepSeekRankTensorRecord> tensor_records,
      Sha256Digest metadata_root) noexcept;

  DeepSeekRankArtifactMetadataBlobFields fields_;
  DeepSeekRankMappingPlan mapping_;
  std::vector<DeepSeekRankTensorRecord> tensor_records_;
  Sha256Digest metadata_root_{};
};

Result<std::vector<std::byte>> encode_deepseek_rank_artifact_metadata_blob(
    const DeepSeekRankArtifactMetadataBlob& blob);
Result<DeepSeekRankArtifactMetadataBlob>
decode_deepseek_rank_artifact_metadata_blob(
    std::span<const std::byte> frame);

}  // namespace pih
