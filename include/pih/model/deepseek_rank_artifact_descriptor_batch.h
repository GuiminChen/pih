#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_transfer_transaction.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankArtifactDescriptorBatchAbi =
    "pih_deepseek_rank_artifact_descriptor_batch_v1";
inline constexpr std::string_view
    kDeepSeekRankArtifactDescriptorBatchFrameAbi =
        "pih_deepseek_rank_artifact_descriptor_batch_frame_v1";
inline constexpr std::size_t
    kDeepSeekRankArtifactDescriptorBatchFrameMaximumBytes = 5616;

struct DeepSeekRankArtifactDescriptorBatchFields final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint32_t batch_index = 0;
  std::uint32_t batch_count = 0;
  std::uint32_t first_descriptor_ordinal = 0;
  std::uint32_t descriptor_count = 0;
  std::uint32_t cumulative_descriptor_count = 0;
  bool final_batch = false;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  Sha256Digest transfer_manifest_root{};
  Sha256Digest artifact_admission_binding_root{};
  Sha256Digest transfer_transaction_root{};
  Sha256Digest descriptor_batch_root{};
  Sha256Digest adopted_descriptor_set_root{};
};

class DeepSeekRankArtifactDescriptorBatch final {
 public:
  static Result<DeepSeekRankArtifactDescriptorBatch> Create(
      DeepSeekRankArtifactDescriptorBatchFields fields,
      std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations);
  static Result<DeepSeekRankArtifactDescriptorBatch> FromView(
      const DeepSeekRankArtifactDescriptorBatchView& view);

  [[nodiscard]] const DeepSeekRankArtifactDescriptorBatchFields& fields()
      const noexcept {
    return fields_;
  }
  [[nodiscard]] std::span<
      const DeepSeekRankArtifactDescriptorExpectation>
  expectations() const noexcept {
    return expectations_;
  }
  [[nodiscard]] const Sha256Digest& frame_root() const noexcept {
    return frame_root_;
  }

 private:
  DeepSeekRankArtifactDescriptorBatch(
      DeepSeekRankArtifactDescriptorBatchFields fields,
      std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations,
      Sha256Digest frame_root) noexcept;

  DeepSeekRankArtifactDescriptorBatchFields fields_;
  std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations_;
  Sha256Digest frame_root_{};
};

Result<std::vector<std::byte>> encode_deepseek_rank_artifact_descriptor_batch(
    const DeepSeekRankArtifactDescriptorBatch& batch);
Result<DeepSeekRankArtifactDescriptorBatch>
decode_deepseek_rank_artifact_descriptor_batch(
    std::span<const std::byte> frame);

}  // namespace pih
